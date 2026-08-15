"""
POST /simulate: run a Monte Carlo comparison of one or more pit
strategies over HTTP. Thin wrapper around monte_carlo.compare_strategies
-- see that module's docstring for what win_probability does (and
doesn't) mean.
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


class StintIn(BaseModel):
    compound: Compound
    laps: int = Field(gt=0)


class StrategyIn(BaseModel):
    name: str
    stints: List[StintIn] = Field(min_length=1)


class SimulateRequest(BaseModel):
    base_laptime: float = Field(
        gt=0, description="Fresh-tire, low-fuel reference lap time, in seconds"
    )
    strategies: List[StrategyIn] = Field(min_length=1)
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


class StrategyResult(BaseModel):
    name: str
    mean: float
    std: float
    p10: float
    p50: float
    p90: float
    win_probability: float
    lap_times: List[float] = Field(
        description="Noise-free lap-by-lap trace (one deterministic run), for charting"
    )
    pit_laps: List[int]


class SimulateResponse(BaseModel):
    results: List[StrategyResult]


@lru_cache(maxsize=1)
def get_tire_model() -> dict:
    """Fit once per process and reuse -- the underlying race CSVs don't change at runtime."""
    return build_tire_model()


@router.post("/simulate", response_model=SimulateResponse)
def simulate(request: SimulateRequest) -> SimulateResponse:
    tire_model = get_tire_model()

    strategies = {
        strategy.name: [Stint(s.compound, s.laps) for s in strategy.stints]
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

    return SimulateResponse(
        results=[
            StrategyResult(
                name=name,
                lap_times=traces[name].lap_times,
                pit_laps=traces[name].pit_laps,
                **stats,
            )
            for name, stats in summary.items()
        ]
    )
