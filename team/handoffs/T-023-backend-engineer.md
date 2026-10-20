# T-023 — backend-engineer — GET /predict/next and GET /predict/history

**Status:** done. Full `pytest`: 303 passed. 9 new tests in `backend/tests/test_api_predict.py`, all mocked, no network.
**Next owner:** qa-engineer, then frontend-engineer (T-024).

## What changed
- `backend/app/services/prediction_service.py` (new): loads `predict_race.py` by file path and reuses its `next_round`, `load`, `_entry_and_grid` and `CONFIG`. The predictor and CLI are untouched, so the numbers are identical (an offline smoke run gave ANT 26.3% win, sums 1.0 and 3.0).
- `backend/app/api/routes/predict.py` (new) and `main.py` (router registered; the existing localhost:3000 CORS applies).
- Full names come from the cached Jolpica raw payloads (`data/jolpica/raw/2026_*_results__o0.json`). If a code is missing, the driver_id is title-cased.
- Cache: one in-memory entry, 10 min TTL, tagged (round, mode, qualifying_published). The tag is known only after the disk-cached data load, so the TTL is what prevents recomputation.
- Failure (any exception or `SystemExit`, e.g. network down or grid mode without quali): serve the newest `data/jolpica/predictions/2026_R*_*.csv` with `stale: true`. If none exists: HTTP 503 `{"detail": "no live prediction available and no saved prediction CSV found"}`.
- Evidence comes from `data/jolpica/eval/walk_forward_2026.csv` (R2-17, chosen config versus its natural baseline: grid order in grid mode, championship order in pre_quali). If the file is missing, `walk_forward` is null.

## API (exact shapes; pydantic models in `routes/predict.py`)
```python
class RaceInfo(BaseModel):  round: int; name: str; circuit: str; date: str; quali_date: str | None
class DriverPrediction(BaseModel):
    code: str; name: str; team: str
    grid: int | None            # null in pre_quali mode; 0 = pit-lane start
    p_win: float; p_podium: float; p_top10: float; exp_pos: float; p_dnf: float   # fractions 0-1
class Headline(BaseModel):  winner: str; podium: list[str]    # podium = top 3 by p_podium
class WalkForward(BaseModel): races: int; win_logloss: float; baseline_win_logloss: float; mode: str  # "grid" | "prequali"
class Evidence(BaseModel):  walk_forward: WalkForward | None; notes: list[str]
class PredictionResponse(BaseModel):
    race: RaceInfo; mode: str            # "pre_quali" | "grid"
    generated_at: str                    # ISO 8601 UTC; for stale = CSV mtime
    stale: bool
    drivers: list[DriverPrediction]      # sorted by p_win desc
    headline: Headline; evidence: Evidence; disclaimer: str

class HistoryRow(BaseModel):
    round: int; race: str
    pole_sitter: str | None              # code of the driver who started 1st
    predicted_winner: str; predicted_winner_p: float
    actual_winner: str; actual_winner_grid: int | None
    p_actual_winner: float; winner_correct: bool
    predicted_podium: list[str]; actual_podium: list[str]; podium_hits: int   # 0-3
class HistoryResponse(BaseModel): season: int; mode: str   # mode always "grid"
                                  races: list[HistoryRow]  # rounds 1-17
```
Real values: `GET /predict/next` currently returns pre_quali, round 18, "United States GP", "Circuit of the Americas", date 2026-10-25, quali_date 2026-10-24. Winner ANT, podium ANT/HAM/LEC. Evidence is 16 races, win log-loss 0.1133 versus a 0.1176 baseline (the grid-mode figures, 0.079 versus 0.0937, appear once qualifying is published). ANT's name is "Andrea Kimi Antonelli", as Jolpica spells it.

## Notes and risks
- The first uncached call runs a 10k-sim prediction (several seconds). Later calls within 10 min are instant. The frontend should show a loading state.
- Grid mode uses the qualifying classification with no penalties, as in the CLI. The disclaimer says so.
- The `walk_forward_2026_by_race.csv` and the eval CSV are static files. They need regenerating (`predict_race.py --by-race`, `--evaluate`) after new races.
- Stale fallback shows `race.name` as "Round N" and a blank circuit if `schedule.csv` is missing.
- Not touched: `tire_model.py`, `optimizer.py`, `BOARD.md`.
