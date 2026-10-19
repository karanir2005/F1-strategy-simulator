"""
GET /tracks and POST /optimize (contract: team/contracts/track-optimizer.md).

Track parameters come from data/tracks/track_params.json via
app.core.track_params; a missing file is a 503, an unknown track_id a 404.
The optimizer itself lives in app.core.optimizer.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.api.deps import resolve_track, tracks_or_503
from app.api.routes.simulation import MAX_RACE_LAPS, get_tire_model
from app.core.monte_carlo import DEFAULT_LAP_TIME_NOISE_STD, DEFAULT_PIT_LOSS_NOISE_STD
from app.core.optimizer import ASSUMPTIONS, NoFeasiblePlan, optimize

router = APIRouter()

MAX_OPT_SIMS = 20_000


class TrackOut(BaseModel):
    """One circuit's parameters (units: seconds, laps). Extra keys in the JSON pass through."""

    model_config = ConfigDict(extra="allow")

    track_id: str
    name: str
    circuit: Optional[str] = None
    total_laps: int
    pit_loss_s: float
    pit_loss_noise_std_s: Optional[float] = None
    base_laptime_s: float
    compounds: Optional[Dict[str, Optional[str]]] = None
    max_stint_laps: Dict[str, int]
    source_season: Optional[int] = None
    source_races: List[str] = []
    on_2026_calendar: Optional[bool] = None
    already_raced_2026: Optional[bool] = None
    notes: Optional[str] = None


class OptimizeRequest(BaseModel):
    track_id: str
    max_stops: int = Field(default=3, ge=1, le=3)
    n_sims: int = Field(default=2000, ge=1, le=MAX_OPT_SIMS)
    top_n: int = Field(default=5, ge=1, le=10)
    total_laps: Optional[int] = Field(default=None, gt=0, le=MAX_RACE_LAPS)
    pit_loss_s: Optional[float] = Field(default=None, ge=0, le=120)
    base_laptime_s: Optional[float] = Field(default=None, gt=0, le=300)
    seed: Optional[int] = 0


class StintOut(BaseModel):
    compound: str
    laps: int
    start_age: int


class StrategyOut(BaseModel):
    rank: int
    n_stops: int
    stints: List[StintOut]
    pit_laps: List[int] = Field(description="Last lap of each stint except the final one (pit at the end of that lap)")
    mean_s: float
    p10_s: float
    p90_s: float
    delta_to_best_mean_s: float
    delta_to_best_p10_s: float
    delta_to_best_p90_s: float


class ParamsUsed(BaseModel):
    total_laps: int
    pit_loss_s: float
    base_laptime_s: float
    n_sims: int


class OptimizeResponse(BaseModel):
    track: TrackOut
    params_used: ParamsUsed
    strategies: List[StrategyOut]
    n_candidates_screened: int
    n_candidates_simulated: int
    assumptions: List[str]


@router.get("/tracks", response_model=List[TrackOut])
def list_tracks() -> List[dict]:
    return tracks_or_503()


@router.post("/optimize", response_model=OptimizeResponse)
def optimize_route(request: OptimizeRequest) -> OptimizeResponse:
    track = TrackOut(**resolve_track(request.track_id))
    total_laps = request.total_laps if request.total_laps is not None else track.total_laps
    pit_loss = request.pit_loss_s if request.pit_loss_s is not None else track.pit_loss_s
    base = request.base_laptime_s if request.base_laptime_s is not None else track.base_laptime_s
    pit_noise = (
        track.pit_loss_noise_std_s if track.pit_loss_noise_std_s is not None else DEFAULT_PIT_LOSS_NOISE_STD
    )
    try:
        result = optimize(
            get_tire_model(),
            total_laps=total_laps,
            pit_loss_s=pit_loss,
            base_laptime_s=base,
            max_stint_laps=track.max_stint_laps,
            max_stops=request.max_stops,
            n_sims=request.n_sims,
            top_n=request.top_n,
            seed=request.seed,
            pit_loss_noise_std=pit_noise,
            lap_time_noise_std=DEFAULT_LAP_TIME_NOISE_STD,
        )
    except NoFeasiblePlan as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return OptimizeResponse(
        track=track,
        params_used=ParamsUsed(
            total_laps=total_laps, pit_loss_s=pit_loss, base_laptime_s=base, n_sims=request.n_sims
        ),
        strategies=result.plans,
        n_candidates_screened=result.n_candidates_screened,
        n_candidates_simulated=result.n_candidates_simulated,
        assumptions=ASSUMPTIONS,
    )
