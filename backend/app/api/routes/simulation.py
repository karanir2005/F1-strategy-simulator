"""
POST /simulate: run a Monte Carlo comparison of one or more pit
strategies over HTTP. Thin wrapper around monte_carlo.compare_strategies
-- see that module's docstring for what win_probability and the paired
delta do (and don't) mean: one car in isolation, additive noise only,
common random numbers across strategies.

Request bounds (all return 422 when violated) exist to keep the
O(n_simulations x laps) noise matrices bounded and to stop inputs that
only produce extrapolated nonsense; see the constants below for why each
number was chosen.
"""

import sys
from functools import lru_cache
from pathlib import Path
from typing import List, Literal, Optional

# make `app.core.*` importable regardless of the caller's cwd (same
# convention used throughout backend/app/core/*.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator

from app.core.monte_carlo import (
    DEFAULT_LAP_TIME_NOISE_STD,
    DEFAULT_N_SIMULATIONS,
    DEFAULT_PIT_LOSS_NOISE_STD,
    compare_strategies,
)
from app.core.race_simulator import DEFAULT_PIT_LOSS_SECONDS, Stint, simulate_race
from app.core.tire_model import build_tire_model

router = APIRouter()

Compound = Literal["SOFT", "MEDIUM", "HARD"]

# Upper bound on a strategy's total laps, and therefore on any single
# stint (a no-stop strategy is one stint of the whole race). The longest
# race in data/processed is Monaco at 78 laps, and F1's minimum race
# distance (~305 km, Monaco excepted) on the shortest circuits keeps
# every real race under ~80 laps, so 100 leaves headroom without
# allowing absurd inputs. This is what bounds memory now that Monte
# Carlo is vectorized: the lap-noise matrix (plus its cumsum/hstack
# copies) is n_simulations x total_laps floats, i.e. at most
# 50,000 x 100 x 8 B = 40 MB each. Unbounded, a large enough `laps`
# raised MemoryError (500) instead of a clean 422 (T-004 QA N-5).
MAX_RACE_LAPS = 100

# Upper bound on StintIn.start_age (tire age on the stint's first lap).
# Real used sets in the 2026 data start at age 2-9, mostly 4-5 (sets
# scrubbed in qualifying; team/research/T-001-fastf1-schema-2026-rules.md).
# 25 leaves ~2.5x headroom for what-ifs (e.g. a set used for a practice
# long run) while keeping the first lap of a used stint inside the tire
# model's fitted age range (max_reliable_age is 37-51 per compound in
# the current fit), instead of deep in the linear extrapolation that made
# start_age=500 return ~322 s laps (T-006 QA F-2/F-3). This does NOT stop
# a long stint from running past max_reliable_age -- that's reachable
# with a fresh set too and is a tire-model question, not an API bound.
MAX_START_AGE = 25

# Bounds the (n_strategies x n_simulations) totals/deltas arrays and the
# per-strategy deterministic runs. The UI compares a handful; 20 is far
# above any realistic comparison.
MAX_STRATEGIES = 20


class StintIn(BaseModel):
    compound: Compound
    laps: int = Field(
        gt=0,
        le=MAX_RACE_LAPS,
        description=f"Laps in this stint, 1..{MAX_RACE_LAPS} (the cap on a whole race)",
    )
    start_age: int = Field(
        default=1,
        ge=1,
        le=MAX_START_AGE,
        description=(
            "Tire age in laps on the first lap of this stint: 1 = new set, "
            ">1 = used set (e.g. 3 for a set scrubbed for 2 laps in qualifying). "
            f"Capped at {MAX_START_AGE}: real 2026 used sets start at age 2-9, and "
            "larger values put the stint in the tire model's extrapolated range, "
            "where lap times stop being meaningful."
        ),
    )


class StrategyIn(BaseModel):
    name: str
    stints: List[StintIn] = Field(min_length=1)


class SimulateRequest(BaseModel):
    base_laptime: float = Field(
        gt=0, description="Fresh-tire, low-fuel reference lap time, in seconds"
    )
    strategies: List[StrategyIn] = Field(
        min_length=1,
        max_length=MAX_STRATEGIES,
        description=(
            f"1..{MAX_STRATEGIES} strategies with unique names, all covering the same "
            f"total laps (sum of stint laps, at most {MAX_RACE_LAPS})"
        ),
    )
    n_simulations: int = Field(default=DEFAULT_N_SIMULATIONS, gt=0, le=50_000)
    lap_time_noise_std: float = Field(default=DEFAULT_LAP_TIME_NOISE_STD, ge=0)
    pit_loss_noise_std: float = Field(default=DEFAULT_PIT_LOSS_NOISE_STD, ge=0)
    pit_loss_seconds: float = Field(default=DEFAULT_PIT_LOSS_SECONDS, ge=0)
    seed: Optional[int] = None

    @field_validator("strategies")
    @classmethod
    def names_must_be_unique(cls, strategies: List[StrategyIn]) -> List[StrategyIn]:
        # compare_strategies keys its internal dict by name, so a
        # duplicate would otherwise silently overwrite an earlier
        # strategy and vanish from the results with no error at all
        names = [s.name for s in strategies]
        if len(names) != len(set(names)):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"strategy names must be unique, duplicated: {dupes}")
        return strategies

    @field_validator("strategies")
    @classmethod
    def total_laps_must_be_bounded(cls, strategies: List[StrategyIn]) -> List[StrategyIn]:
        too_long = {
            s.name: total
            for s in strategies
            if (total := sum(st.laps for st in s.stints)) > MAX_RACE_LAPS
        }
        if too_long:
            raise ValueError(
                f"total race laps (sum of stint laps) must be at most {MAX_RACE_LAPS}, "
                f"got {too_long}"
            )
        return strategies

    @field_validator("strategies")
    @classmethod
    def total_laps_must_match(cls, strategies: List[StrategyIn]) -> List[StrategyIn]:
        # Under common random numbers a shorter race shares the longer
        # one's first-N lap draws and just skips the rest, so it "wins"
        # every iteration -- comparing different race lengths is
        # meaningless. compare_strategies raises too; checking here gives
        # the client a 422 that names each strategy's total.
        totals = {s.name: sum(st.laps for st in s.stints) for s in strategies}
        if len(set(totals.values())) > 1:
            raise ValueError(
                "all strategies must cover the same total race laps (sum of stint laps) "
                f"to be compared, got {totals}"
            )
        return strategies


_CAVEAT = (
    "Model caveats: one car in isolation (no traffic, safety cars or opponents), "
    "additive iid lap/pit noise, and common random numbers (every strategy sees "
    "the same draws in an iteration)."
)


class StrategyResult(BaseModel):
    name: str
    mean: float = Field(description="Mean total race time, seconds")
    std: float = Field(description="Std of total race time across simulations, seconds")
    p10: float = Field(description="10th percentile total race time, seconds")
    p50: float = Field(description="Median total race time, seconds")
    p90: float = Field(description="90th percentile total race time, seconds")
    win_probability: float = Field(
        description=(
            "Share (0..1) of simulations in which this strategy had the lowest total "
            "among the strategies in this request (exact ties go to the first listed). "
            "NOT a race-win chance. " + _CAVEAT + " Under these, strategies with the "
            "same stop count differ by a constant every iteration, so this is 0 or 1 "
            "even for a sub-second gap -- prefer delta_mean/delta_p10/delta_p90."
        )
    )
    delta_mean: float = Field(
        description=(
            "Paired gap to the reference strategy (SimulateResponse.reference_strategy), "
            "seconds: mean over simulations of (this total - reference total) in the "
            "same simulation. Positive = slower than the reference; exactly 0 for the "
            "reference itself. " + _CAVEAT + " A gap under this model, not a "
            "prediction of real race gaps (replay errors are tens of seconds)."
        )
    )
    delta_p10: float = Field(
        description=(
            "10th percentile of the same per-simulation paired gap, seconds "
            "(0 for the reference). " + _CAVEAT
        )
    )
    delta_p90: float = Field(
        description=(
            "90th percentile of the same per-simulation paired gap, seconds "
            "(0 for the reference). " + _CAVEAT
        )
    )
    lap_times: List[float] = Field(
        description="Noise-free lap-by-lap trace (one deterministic run), for charting"
    )
    pit_laps: List[int]


class SimulateResponse(BaseModel):
    reference_strategy: str = Field(
        description=(
            "Name of the strategy every delta_* is measured against: the one with the "
            "lowest mean total time (ties -> first listed)"
        )
    )
    results: List[StrategyResult]


@lru_cache(maxsize=1)
def get_tire_model() -> dict:
    """Fit once per process and reuse -- the underlying race CSVs don't change at runtime."""
    return build_tire_model()


@router.post("/simulate", response_model=SimulateResponse)
def simulate(request: SimulateRequest) -> SimulateResponse:
    tire_model = get_tire_model()

    strategies = {
        strategy.name: [Stint(s.compound, s.laps, s.start_age) for s in strategy.stints]
        for strategy in request.strategies
    }

    summary = compare_strategies(
        strategies,
        base_laptime=request.base_laptime,
        tire_model=tire_model,
        n_simulations=request.n_simulations,
        lap_time_noise_std=request.lap_time_noise_std,
        pit_loss_noise_std=request.pit_loss_noise_std,
        pit_loss_seconds=request.pit_loss_seconds,
        seed=request.seed,
    )

    # one noise-free run per strategy purely for the lap-by-lap chart --
    # the Monte Carlo stats above are what should be trusted for
    # comparisons, this is just a representative shape to look at
    traces = {
        name: simulate_race(
            stints,
            base_laptime=request.base_laptime,
            tire_model=tire_model,
            pit_loss_seconds=request.pit_loss_seconds,
        )
        for name, stints in strategies.items()
    }

    # every entry carries the same delta_reference; lift it to the top level
    reference = next(iter(summary.values()))["delta_reference"]

    return SimulateResponse(
        reference_strategy=reference,
        results=[
            StrategyResult(
                name=name,
                lap_times=traces[name].lap_times,
                pit_laps=traces[name].pit_laps,
                **{k: v for k, v in stats.items() if k != "delta_reference"},
            )
            for name, stats in summary.items()
        ],
    )
