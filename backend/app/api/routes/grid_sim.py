"""
POST /simulate/grid (T-034): full-grid multi-car race simulation. Thin wrapper over
app.core.race_grid_sim + app.core.grid_inputs; read race_grid_sim's docstring for the model and what it
leaves out. /simulate (one car, clean air) and /optimize are unchanged.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.api.deps import resolve_track
from app.api.routes.simulation import MAX_RACE_LAPS, StintIn, get_tire_model
from app.core import grid_inputs as G
from app.core.race_grid_sim import load_interaction_params, simulate_grid
from app.core.race_simulator import Stint
from app.core.safety_cars import load_params as load_safety_car_params

router = APIRouter()

MAX_GRID_SIMS = 20_000


class GridEntry(BaseModel):
    code: str = Field(description="Three-letter driver code, e.g. NOR")
    grid: int = Field(ge=0, le=30, description="Starting slot; 0 = pit-lane start (placed last)")


class GridSimRequest(BaseModel):
    track_id: str = Field(description="Track from GET /tracks")
    round: Optional[int] = Field(
        default=None, ge=1, le=30,
        description="2026 round for the entry list, qualifying and DNF rates. Default: the round at track_id.",
    )
    grid: Optional[List[GridEntry]] = Field(
        default=None,
        description="Optional starting grid override (every driver of the entry list). Default: the race's real "
        "grid if it has been run, else the latest qualifying classification (no penalties applied).",
    )
    n_sims: int = Field(default=2000, ge=1, le=MAX_GRID_SIMS)
    seed: Optional[int] = Field(default=0, ge=0)
    safety_cars: bool = Field(default=True, description="Random fitted SC/VSC periods with field bunching")
    forced_sc_lap: Optional[int] = Field(default=None, ge=1, le=MAX_RACE_LAPS, description="Force an SC starting on this lap")
    forced_sc_laps: Optional[int] = Field(default=None, ge=1, le=20, description="Length of the forced SC (default: median fitted length)")
    strategies: Optional[Dict[str, List[StintIn]]] = Field(
        default=None,
        description="Override {driver code: stints} for those cars; every other car keeps the default "
        "(optimizer top plans with per-car variation). Each must sum to the race laps.",
    )
    total_laps: Optional[int] = Field(default=None, gt=0, le=MAX_RACE_LAPS)


class StintOut(BaseModel):
    compound: str
    laps: int
    start_age: int


class DriverOut(BaseModel):
    code: str
    team: str
    grid: int
    pace_delta_s: float = Field(description="Assumed seconds per lap slower than the quickest car (input)")
    p_dnf_input: float
    strategy: List[StintOut]
    p_win: float
    p_podium: float
    p_top10: float
    p_dnf: float
    exp_points: float
    mean_position: float
    p10_time_s: Optional[float] = Field(description="Race time percentiles over finishing runs, seconds (null if never finishes)")
    p50_time_s: Optional[float]
    p90_time_s: Optional[float]
    position_probs: List[float] = Field(description="P(finish in position 1..n); retirements rank last")


class GridSimResponse(BaseModel):
    track_id: str
    round: int
    total_laps: int
    n_sims: int
    seed: Optional[int]
    grid_source: str
    drivers: List[DriverOut] = Field(description="Sorted by expected points, best first")
    mean_overtakes_per_race: float
    mean_sc_laps_per_race: float
    interaction_source: str
    interaction_placeholder_keys: List[str]
    forced_sc_lap: Optional[int]
    assumptions: List[str]


ASSUMPTIONS = [
    "Pace comes from qualifying (Q1 gaps, shrunk to the team mean) blended with earlier-race ratings; the scale "
    "from quali gap to race pace, the shrinkage and blend weights, the pace noise, SC bunching, pass rate and strategy "
    "mix are fitted on 19 dry 2026 races (T-060, data/sim_params/grid_sim_calibration.json; SC bunching and the "
    "strategy mix rest on thin evidence). The pace spread is a 2026 fact and did not carry over to 2025.",
    "Strategies are fixed in advance per car (default: optimizer top plans with per-car stop jitter); no reactive stops "
    "here (see POST /strategy/policy for the safety-car rule).",
    "Finishing-order skill is about that of the grid order and the Plackett-Luce predictor (T-060); the sim's win "
    "probabilities are more diffuse than the predictor's. Use it for strategy what-ifs, not as a winner forecast.",
    "No lapping/blue flags, red flags, weather, pit-lane queues, penalties or DNF-triggered safety cars.",
    "Dirty air, overtake probability, pass costs and the lap-1 shuffle are the T-053 fits (2026, 19 races).",
    "Validation vs the grid order and the Plackett-Luce predictor: see team/handoffs/T-034-backend-engineer.md.",
]


def _nan_none(x: float) -> Optional[float]:
    return None if x != x else float(x)


@router.post("/simulate/grid", response_model=GridSimResponse)
def simulate_grid_route(request: GridSimRequest) -> GridSimResponse:
    track = resolve_track(request.track_id)
    tire_model = get_tire_model()
    grid = {e.code: e.grid for e in request.grid} if request.grid else None
    strategies = (
        {c: [Stint(s.compound, s.laps, s.start_age) for s in v] for c, v in request.strategies.items()}
        if request.strategies
        else None
    )
    try:
        gi = G.build_inputs(
            request.track_id,
            tire_model,
            rnd=request.round,
            grid=grid,
            strategies=strategies,
            total_laps=request.total_laps,
            seed=request.seed or 0,
        )
        ip = load_interaction_params()
        sim = simulate_grid(
            gi.cars,
            gi.track["base_laptime_s"],
            tire_model,
            gi.total_laps,
            n_sims=request.n_sims,
            seed=request.seed,
            pit_loss_s=gi.track["pit_loss_s"],
            interaction=ip,
            safety_cars=load_safety_car_params() if request.safety_cars else None,
            forced_sc_lap=request.forced_sc_lap,
            forced_sc_laps=request.forced_sc_laps,
            track_id=request.track_id,
        )
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    by_code = {c.code: c for c in gi.cars}
    drivers = []
    for r in sorted(sim.summary(), key=lambda r: (-r["exp_points"], r["mean_position"])):
        car = by_code[r["code"]]
        drivers.append(
            DriverOut(
                **{k: v for k, v in r.items() if k not in ("p10_time_s", "p50_time_s", "p90_time_s")},
                pace_delta_s=car.pace_delta_s,
                p_dnf_input=car.p_dnf,
                strategy=[StintOut(compound=s.compound, laps=s.laps, start_age=s.start_age) for s in car.strategy],
                p10_time_s=_nan_none(r["p10_time_s"]),
                p50_time_s=_nan_none(r["p50_time_s"]),
                p90_time_s=_nan_none(r["p90_time_s"]),
            )
        )
    res = G.load_history()["results"]
    raced = bool(((res["season"] == G.SEASON) & (res["round"] == gi.rnd)).any())
    src = "request grid override" if grid else ("race grid (race already run)" if raced else "qualifying classification (no penalties)")
    return GridSimResponse(
        track_id=track["track_id"],
        round=gi.rnd,
        total_laps=gi.total_laps,
        n_sims=request.n_sims,
        seed=request.seed,
        grid_source=src,
        drivers=drivers,
        mean_overtakes_per_race=float(sim.n_overtakes.mean()),
        mean_sc_laps_per_race=float(sim.n_sc_laps.mean()),
        interaction_source=ip.source,
        interaction_placeholder_keys=ip.placeholder_keys,
        forced_sc_lap=request.forced_sc_lap,
        assumptions=ASSUMPTIONS + gi.notes,
    )
