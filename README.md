# F1 Strategy Simulator

A full-stack Formula 1 race strategy simulator and race predictor. A Monte Carlo engine evaluates pit strategies from tyre, fuel and pit-stop models fitted on real 2026 race data, and a ranking model predicts race and qualifying results, validated walk-forward across the 2026 season.

## Highlights

- **Monte Carlo strategy simulation**: compound-specific tyre wear with stint-length durability, fuel burn-off, pit-stop time loss, and optional safety cars / VSCs. Vectorized with NumPy and run with common random numbers, so strategies are compared on paired deltas instead of independent noisy runs.
- **Strategy optimizer**: finds optimal stint plans from per-track parameters (race laps, pit loss, compounds, p90 stint caps). Benchmarked against what teams actually ran: across 14 races, the most common real top-10 strategy was in the optimizer's top 3 plans in 11 of them.
- **Calibrated, not assumed**: tyre curves, pit loss, lap-time noise and safety-car rates are fitted from real lap data. Safety-car laps and wet races are filtered out of the tyre fit.
- **Race and qualifying predictor**: a Plackett-Luce ranking model with a down-weighted 2025 prior, per-circuit "grid stickiness" from pole-to-win history and a DNF model. It outputs P(win), P(podium), P(top 10), plus pole, front row, Q3 and expected grid.
- **Results so far (2026, rounds 2–20)**: with the grid set, win log-loss is 0.076 vs 0.098 for the grid-order baseline. The model called 14 of 20 race winners and 37 of 60 podium places.

## Tech stack

- **Backend:** Python, FastAPI, NumPy, pandas, SciPy, pydantic
- **Data:** FastF1 (cached sessions), [OpenF1](https://openf1.org) (laps, stints, race control, weather, pit stops), [Jolpica-F1](https://github.com/jolpica/jolpica-f1) (results, qualifying, grids)
- **Frontend:** Next.js 14, Tailwind CSS, shadcn/ui, Recharts, Motion

## Project structure

```
backend/app/
├── main.py                 # FastAPI entrypoint
├── core/
│   ├── data_loader.py      # FastF1 session -> per-lap DataFrame
│   ├── openf1_loader.py    # OpenF1 ingestion (laps, stints, race control, weather)
│   ├── jolpica_client.py   # Jolpica results / qualifying / grids
│   ├── tire_model.py       # tyre degradation + fuel-burn fits
│   ├── race_simulator.py   # lap-by-lap race time for one strategy
│   ├── monte_carlo.py      # N noisy runs -> distributions, paired deltas
│   ├── safety_cars.py      # SC / VSC sampling
│   ├── optimizer.py        # optimal stint plans
│   ├── track_params.py     # per-track parameters
│   └── predictor.py        # race + qualifying outcome model
├── api/routes/             # /simulate, /optimize, /tracks, /predict/*
└── services/               # prediction service
backend/tests/              # pytest suite
frontend/                   # Next.js app: /predict and /strategy pages
data/                       # processed race data, track params, eval outputs
```

Repo-root scripts handle data pulls (`pull_races.py`, `pull_openf1.py`), calibration (`calibrate_*.py`, `build_track_params.py`), validation (`validate_replay.py`, `validate_scorecard.py`) and prediction (`predict_race.py`).

## Getting started

```powershell
# Backend (Python 3.14)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt

cd backend
uvicorn app.main:app --reload       # http://127.0.0.1:8000
```

```powershell
# Frontend
cd frontend
npm install
npm run dev                         # http://localhost:3000
```

```powershell
# Tests (from repo root)
pytest

# Race prediction from the command line
python predict_race.py              # next race; grid mode once qualifying is out
python predict_race.py --evaluate   # walk-forward 2026 vs baselines
```

## API

| Method | Route | Purpose |
|---|---|---|
| `POST` | `/simulate` | Monte Carlo comparison of user-defined strategies |
| `POST` | `/optimize` | Optimal stint plans for a track |
| `GET` | `/tracks` | Available tracks and their parameters |
| `GET` | `/predict/next` | Prediction for the next race (pre-quali or grid mode) |
| `GET` | `/predict/history` | Season-to-date prediction accuracy |
| `GET` | `/predict/races`, `/predict/race/{round}` | Past races: prediction vs actual |

## Validation

- **Replays:** `validate_replay.py` re-simulates real drivers' strategies. Error is 0.13–1.10% of total race time, excluding SC/VSC/red-flag laps.
- **Leak-free scorecard:** `validate_scorecard.py` uses leave-one-race-out tyre fits and qualifying-based pace.
- **Predictor:** evaluated walk-forward. Each race is predicted using only earlier races and compared against uniform, grid-order, championship-order and recent-form baselines.

## Limitations

- **One car at a time.** The simulator models a single car in isolation, without traffic or overtaking. A full-grid multi-car simulation is the next planned step.
- **Data sources.** FastF1 live timing has required an F1 TV subscription since October 2026, so new races come from OpenF1 and Jolpica.
- **Local only.** The app is not deployed yet and runs locally.
