# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

F1 race strategy simulator with a Monte Carlo engine. Models tire degradation, pit stop timing, safety cars, overtaking probabilities, and weather effects. Phase 2 adds an XGBoost ML model to predict race winners.

**Current state:** Phase 1 core simulation engine is built and validated: `data_loader.py`, `tire_model.py`, `race_simulator.py`, `monte_carlo.py`, and the `POST /simulate` API route all exist (see Build Order below). `backend/app/services/` holds the prediction service. The Next.js frontend in `frontend/` has two pages: `/predict` (race winner/podium prediction) and `/strategy` (track picker, optimal stint plans, manual comparison).

**Next (see `team/BOARD.md`):** v1.0.0 work is done except deployment (T-032, deferred; the app runs locally). Backlog: T-033 (copy an optimizer plan into the manual comparison with one click) and T-034 (Phase 1 full-grid multi-car race simulation, the final polishing step after v1.0.0). Why the engine can't yet rank strategies beyond the grid baseline is in `team/ROADMAP.md`.

## Agent team

Work is split across subagents in `.claude/agents/` (researcher, data-scientist, backend-engineer, frontend-engineer, qa-engineer). The main session orchestrates. Protocol, lane ownership and the handoff template are in `team/README.md`, tasks are in `team/BOARD.md` (orchestrator-only edits), and handoffs and research notes go in `team/handoffs/` and `team/research/`. Every code task goes through `qa-engineer` before it's marked done.

## Running the Project

```powershell
# Activate virtual environment (Python 3.14.2)
.venv\Scripts\Activate.ps1

# Install dependencies (requirements-dev.txt also pulls in requirements.txt)
pip install -r requirements-dev.txt

# Test the data pipeline
python test_data_load.py

# Run the API (from backend/)
cd backend && uvicorn app.main:app --reload

# Run tests (from repo root)
pytest
```

Dependencies are pinned in `requirements.txt` (runtime: fastf1, pandas, numpy, fastapi, uvicorn, pydantic) and `requirements-dev.txt` (adds pytest, httpx for testing). If you add a new dependency, install it and add it to the appropriate file with its pinned version.

FastF1 caches data in `cache/` — first load of a session is slow (network download), subsequent loads are fast. Do not delete this directory.

## Architecture

```
backend/app/
├── main.py          # FastAPI app entrypoint (uvicorn app.main:app)
├── core/
│   ├── data_loader.py     # FastF1 session -> per-lap DataFrame (incl. is_green_flag); also extract_pit_stops() for real pit-lane durations
│   ├── tire_model.py      # fits degradation + fuel-burn curves from data/processed/*.csv
│   ├── race_simulator.py  # single-strategy lap loop -> total race time
│   └── monte_carlo.py     # N noisy runs of race_simulator -> distribution + win_probability
├── api/routes/
│   └── simulation.py      # POST /simulate
└── services/        # empty — not yet needed
data/processed/     # Output CSVs from data pipeline (one per pulled race)
cache/              # FastF1 local cache (do not delete)
```

**Entry point for the data pipeline:** [test_data_load.py](test_data_load.py) → calls `get_race_data(year, gp)` from `backend/app/core/data_loader.py`, which loads a FastF1 race session, extracts per-lap driver/compound/tire-age/pit/track-status data into a DataFrame, and writes it to `data/processed/`. [pull_races.py](pull_races.py) does this for a whole list of races at once (used to build the tire model's training set).

**Entry point for the API:** `backend/app/main.py` (run with `uvicorn app.main:app --reload` from `backend/`).

**Validation:** [validate_replay.py](validate_replay.py) reconstructs a real driver's actual strategy from pulled race data, runs it through `race_simulator.py`, and compares simulated vs. actual total race time. Rerun this after any change to `tire_model.py` or `race_simulator.py` to check nothing regressed.

## Tech Stack

- **Backend:** Python + FastAPI + FastF1 + numpy/pandas/scipy + pydantic
- **Frontend:** Next.js 14 + Tailwind CSS + shadcn/ui + Recharts + D3.js
- **Data sources:** FastF1 (cached 2026 races; live timing is paywalled), OpenF1, Jolpica-F1 API (not Ergast — it's deprecated)
- **Infra (planned):** PostgreSQL, Redis, Docker Compose, Railway/Render + Vercel

## Build Order (per project plan)

The 4 core modules, all built:

1. `backend/app/core/tire_model.py` — compound-specific polynomial degradation curves + cliff effects. Fitted from `data/processed/*_laps.csv` (currently 16 races, 2026-only by default). See the module docstring for the safety-car-contamination bug that was found and fixed via `is_green_flag` filtering, and for `fit_shared_fuel_coef`'s known remaining gap. Wear is constrained SOFT >= MEDIUM >= HARD and pace offsets are now fitted (the free 2026 fit is inverted, so the ordering is forced); `COMPOUND_CLIFF_PENALTY_PLACEHOLDER` is still a placeholder — see their comments for why.
2. `backend/app/core/race_simulator.py` — lap loop with tire degradation, fuel burn-off, and pit stop time loss. `DEFAULT_PIT_LOSS_SECONDS` (23.5s) is calibrated from real pit-lane duration data — see `calibrate_pit_loss.py` and the constant's comment.
3. `backend/app/core/monte_carlo.py` — N simulation runs with noise in lap time and pit duration; both noise magnitudes (`DEFAULT_LAP_TIME_NOISE_STD` 0.8s, `DEFAULT_PIT_LOSS_NOISE_STD` 5.0s) are calibrated from real data — see `calibrate_lap_noise.py` / `calibrate_pit_loss.py` and the module docstring for the scope note on what `win_probability` does and doesn't mean.
4. `backend/app/api/routes/simulation.py` — `POST /simulate` endpoint, wired into `backend/app/main.py`.

Simulation parameters (degradation rates, pit loss, safety car frequency) should be calibrated from real FastF1 data via `data_loader.py`, not assumed — treat any parameter numbers from early planning notes as unvalidated starting guesses, not targets. Some are still placeholders; grep for "placeholder" across `backend/app/core/` to find them before trusting strategy comparisons in a real decision. `calibrate_pit_loss.py` and `calibrate_lap_noise.py` (repo root) are the calibration scripts for the ones that have been fitted — re-run them if the pulled race set in `pull_races.py` changes materially.

The simulation was validated against historical race replays (`validate_replay.py`) before the API layer was added, per plan — current error across 5 replays is 0.13–1.10% excluding SC/VSC/red laps (the engine has no safety cars yet), 0.0–4.7% including them. Note `validate_replay.py` uses the driver's own fastest lap as `base_laptime`, which leaks the answer; the non-leaky check is `validate_scorecard.py` (leave-one-race-out tyre fits, qualifying-based pace, results in `data/scorecard/latest.json`). Tests live in `backend/tests/` (`pytest` from repo root).

**Data access:** FastF1 live timing has been paywalled since 2026-10-15 and FastF1 silently returns empty data, so new races come from OpenF1 (`api.openf1.org`) and Jolpica instead. See the DATA ACCESS INCIDENT section in `team/BOARD.md`. The race predictor (`predict_race.py`, `backend/app/core/predictor.py`) runs on Jolpica data.

## Key Design Decisions

- **DP + Genetic Algorithm** for pit strategy optimization (DP for optimal windows, GA for multi-stop search via DEAP)
- **XGBoost over neural networks** for ML predictor — better interpretability and handles F1's sparse historical data
- **Time-series cross-validation** is mandatory for ML — never use future race data to predict past races
- **Redis** for hot cache during Monte Carlo iterations (FastF1 fetching is slow)
- Monte Carlo default: **10,000 simulations** per strategy; output mean position, win probability, p10/p50/p90 race times
