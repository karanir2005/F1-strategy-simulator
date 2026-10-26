"""GET /predict/next and GET /predict/history (T-023). Contract: team/contracts/prediction-api.md."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import prediction_service as svc

router = APIRouter(prefix="/predict", tags=["predict"])


class RaceInfo(BaseModel):
    round: int
    name: str
    circuit: str
    date: str
    quali_date: str | None = None
    race_start_utc: str | None = None  # ISO UTC, e.g. 2026-10-25T19:00:00Z; null if schedule lacks a time
    quali_start_utc: str | None = None


class ActualResult(BaseModel):
    position: int | None = None  # official classification position (incl. classified-but-lapped and DNF slots)
    classified: bool | None = None
    status: str | None = None  # Jolpica status text, e.g. "Finished", "+1 Lap", "Engine"
    dnf: bool = False


class RaceListItem(BaseModel):
    round: int
    name: str
    date: str
    has_prediction: bool
    is_next: bool = False


class DriverPrediction(BaseModel):
    code: str
    name: str
    team: str
    grid: int | None = None  # null in pre_quali mode; 0 = pit-lane start
    p_win: float
    p_podium: float
    p_top10: float
    exp_pos: float
    p_dnf: float
    # T-029, grid mode only (absent in pre_quali). Previously dropped by this response model.
    p_win_prequali: float | None = None
    p_podium_prequali: float | None = None
    actual: ActualResult | None = None  # T-044, only on /predict/race/{round}


class QualiDriver(BaseModel):
    code: str
    p_pole: float
    p_front_row: float
    p_q3: float
    p_q1_exit: float
    exp_grid: float
    actual_grid: int | None = None  # grid mode only; 0 = pit-lane start


class QualiEvidence(BaseModel):
    races: int
    pole_logloss: float
    baseline_pole_logloss: float
    baseline_name: str
    spearman: float
    baseline_spearman: float
    q3_logloss: float | None = None
    baseline_q3_logloss: float | None = None


class Qualifying(BaseModel):
    available: bool
    drivers: list[QualiDriver]
    evidence: QualiEvidence | None = None


class Headline(BaseModel):
    winner: str
    podium: list[str]


class WalkForward(BaseModel):
    races: int
    win_logloss: float
    baseline_win_logloss: float
    mode: str


class Evidence(BaseModel):
    walk_forward: WalkForward | None = None
    notes: list[str]


class PredictionResponse(BaseModel):
    race: RaceInfo
    mode: str  # "pre_quali" | "grid"
    generated_at: str
    stale: bool
    stale_reason: str | None = None  # why the live path failed when stale=true
    # "qualifying_classification" | "override_file" | null (pre_quali, or unknown for an old saved CSV)
    grid_source: str | None = None
    drivers: list[DriverPrediction]
    qualifying: Qualifying | None = None  # T-035
    headline: Headline
    evidence: Evidence
    disclaimer: str
    race_done: bool | None = None  # T-044, true only on /predict/race/{round}


class HistoryRow(BaseModel):
    round: int
    race: str
    pole_sitter: str | None = None
    predicted_winner: str
    predicted_winner_p: float
    actual_winner: str
    actual_winner_grid: int | None = None
    p_actual_winner: float
    winner_correct: bool
    predicted_podium: list[str]
    actual_podium: list[str]
    podium_hits: int


class HistoryResponse(BaseModel):
    season: int
    mode: str
    races: list[HistoryRow]


@router.get("/next", response_model=PredictionResponse, response_model_exclude_unset=True)
def predict_next() -> dict:
    try:
        return svc.get_next()
    except svc.PredictionUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/history", response_model=HistoryResponse)
def predict_history() -> dict:
    try:
        return {"season": svc.SEASON, "mode": "grid", "races": svc.history()}
    except svc.PredictionUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/races", response_model=list[RaceListItem])
def predict_races() -> list[dict]:
    return svc.list_races()


@router.get("/race/{round}", response_model=PredictionResponse, response_model_exclude_unset=True)
def predict_race_round(round: int) -> dict:
    try:
        return svc.get_past_race(round)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"no precomputed prediction for round {round}")
