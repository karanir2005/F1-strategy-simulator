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
from app.core.optimizer import ASSUMPTIONS, SAFETY_CAR_ASSUMPTION, NoFeasiblePlan, optimize
from app.core.safety_cars import load_params as load_safety_car_params

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
    # optional: p90 of observed stint lengths per compound (null where < 5 stints); optimizer cap = min(max, typical)
    typical_stint_laps: Optional[Dict[str, Optional[int]]] = None
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
    seed: Optional[int] = Field(default=0, ge=0)
    use_track_wear: bool = Field(
        default=False,
        description="T-030: use this track's per-track tyre wear fit (T-028) instead of the global curves. "
        "Falls back to global if the track has no fit; params_used.use_track_wear says what was applied.",
    )
    safety_cars: bool = Field(
        default=False,
        description="T-030: add random SC/VSC periods (fitted on 2025-26 races), same timeline for every plan.",
    )


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
    # optional (T-021): stint cap per compound actually used (observed max, floored to 15, clipped to race laps)
    effective_max_stint_laps: Optional[Dict[str, int]] = None
    min_stint_laps: Optional[int] = None
    # optional (T-030): what was actually applied
    use_track_wear: Optional[bool] = None
    safety_cars: Optional[bool] = None


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
    tire_model = get_tire_model()
    wear_applied = bool(request.use_track_wear and request.track_id in tire_model.get("_track_wear", {}))
    try:
        result = optimize(
            tire_model,
            total_laps=total_laps,
            pit_loss_s=pit_loss,
            base_laptime_s=base,
            max_stint_laps=track.max_stint_laps,
            typical_stint_laps=track.typical_stint_laps,
            max_stops=request.max_stops,
            n_sims=request.n_sims,
            top_n=request.top_n,
            seed=request.seed,
            pit_loss_noise_std=pit_noise,
            lap_time_noise_std=DEFAULT_LAP_TIME_NOISE_STD,
            track_id=request.track_id if request.use_track_wear else None,
            safety_cars=load_safety_car_params() if request.safety_cars else None,
        )
    except NoFeasiblePlan as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return OptimizeResponse(
        track=track,
        params_used=ParamsUsed(
            total_laps=total_laps, pit_loss_s=pit_loss, base_laptime_s=base, n_sims=request.n_sims,
            effective_max_stint_laps=result.effective_caps, min_stint_laps=result.min_stint_laps,
            use_track_wear=wear_applied, safety_cars=request.safety_cars,
        ),
        strategies=result.plans,
        n_candidates_screened=result.n_candidates_screened,
        n_candidates_simulated=result.n_candidates_simulated,
        assumptions=[
            SAFETY_CAR_ASSUMPTION if (request.safety_cars and a.startswith("No safety cars")) else a
            for a in ASSUMPTIONS
            if not (wear_applied and a.startswith("Global tyre-wear"))
        ]
        + (["Per-track tyre wear slopes (partially pooled to the global fit) used for this circuit."] if wear_applied else []),
    )
