# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

F1 race strategy simulator with a Monte Carlo engine. Models tire degradation, pit stop timing, safety cars, overtaking probabilities, and weather effects. Phase 2 adds an XGBoost ML model to predict race winners.

**Current state:** Phase 1 in progress — data loading module exists, simulation engine not yet built.

## Running the Project

```powershell
# Activate virtual environment (Python 3.14.2)
.venv\Scripts\Activate.ps1

# Test the data pipeline
python test_data_load.py
```

FastF1 caches data in `cache/` — first load of a session is slow (network download), subsequent loads are fast.

## Architecture

```
backend/app/
├── core/           # Simulation logic (data_loader.py exists; tire_model, race_simulator, monte_carlo to be built)
├── api/            # FastAPI routes (empty — to be built)
└── services/       # Service wrappers (empty — to be built)
data/processed/     # Output CSVs from data pipeline
cache/              # FastF1 local cache (do not delete)
```

**Entry point for the data pipeline:** [test_data_load.py](test_data_load.py) → calls `get_race_data(year, gp)` from `backend/app/core/data_loader.py`.

## Tech Stack

- **Backend:** Python + FastAPI + FastF1 + numpy/pandas/scipy + pydantic
- **Frontend (planned):** Next.js 14 + Tailwind CSS + shadcn/ui + Recharts + D3.js
- **Data sources:** FastF1 (primary), Jolpica-F1 API (not Ergast — it's deprecated)
- **Infra (planned):** PostgreSQL, Redis, Docker Compose, Railway/Render + Vercel

## Build Order (per project plan)

The 4 core modules to build next, in order:

1. `backend/app/core/tire_model.py` — compound-specific polynomial degradation curves + cliff effects
2. `backend/app/core/race_simulator.py` — lap loop with pit stop time loss
3. `backend/app/core/monte_carlo.py` — N simulation runs with noise in degradation + pit duration
4. `backend/app/api/routes/simulation.py` — `POST /simulate` endpoint

Then add frontend only after the simulation engine is validated against historical data.

## Key Design Decisions

- **DP + Genetic Algorithm** for pit strategy optimization (DP for optimal windows, GA for multi-stop search via DEAP)
- **XGBoost over neural networks** for ML predictor — better interpretability and handles F1's sparse historical data
- **Time-series cross-validation** is mandatory for ML — never use future race data to predict past races
- **Redis** for hot cache during Monte Carlo iterations (FastF1 fetching is slow)
- Monte Carlo default: **10,000 simulations** per strategy; sample tire deg variance ±10%, pit stop time μ=22s σ=1.5s, SC events via Poisson process

## Simulation Parameters (reference)

- Tire degradation rates: SOFT ~0.08–0.12s/lap, MEDIUM ~0.04–0.07s/lap, HARD ~0.02–0.04s/lap
- Safety car probability: ~1.5% per lap (SC), ~1% per lap (VSC)
- Pit stop time loss: μ=22s, σ=1.5s
- Target Monte Carlo output: mean position, win probability, p10/p50/p90 race times
