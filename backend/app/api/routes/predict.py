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
    drivers: list[DriverPrediction]
    headline: Headline
    evidence: Evidence
    disclaimer: str


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


@router.get("/next", response_model=PredictionResponse)
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
