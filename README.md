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

## In-app chat (local, optional)

The `/strategy` page has an "Ask about this strategy" panel backed by `POST /chat`, which runs a tool-calling loop (max 5 rounds) against a local [Ollama](https://ollama.com) model, using the same tools as the MCP server. No API key.

```powershell
# install Ollama, then:
ollama pull qwen2.5:7b
```

Override with `OLLAMA_URL` (default `http://localhost:11434`) and `OLLAMA_MODEL` (default `qwen2.5:7b`). `GET /chat/status` reports whether Ollama and the model are available; `POST /chat` returns 503 if not. This is local only: the deployed free-tier backend has no Ollama, so the panel shows the setup note there. A 7B model on CPU can take a minute or more per answer and can be wrong; the panel shows which tools it used and the caveats.

## MCP server (chat with the model in Claude)

`backend/app/mcp_server.py` is a stdio MCP server that exposes the simulator to Claude Desktop or Claude Code using your Claude subscription (no API key, no uvicorn needed). Tools: `list_tracks`, `predict_race`, `get_race_results`, `optimize_strategy`, `simulate_strategy`, `query_laps`, and `search_project_notes` (BM25 retrieval over `team/` notes and README; Claude writes the answer from the retrieved chunks). Every result includes a `caveats` list.

**Claude Code:** the project-scope `.mcp.json` at the repo root is picked up automatically (approve it on first use).

**Claude Desktop:** add this to `%APPDATA%\Claude\claude_desktop_config.json` and restart Claude Desktop:

```json
{
  "mcpServers": {
    "f1-strategy-simulator": {
      "command": "C:\\Users\\rushi\\Desktop\\Folders\\Code\\F1-strategy-simulator\\.venv\\Scripts\\python.exe",
      "args": ["-m", "app.mcp_server"],
      "cwd": "C:\\Users\\rushi\\Desktop\\Folders\\Code\\F1-strategy-simulator\\backend"
    }
  }
}
```

Spot-check questions with expected tools and facts: `team/research/T-054-mcp-eval.md`.

## Validation

- **Replays:** `validate_replay.py` re-simulates real drivers' strategies. Error is 0.13–1.10% of total race time, excluding SC/VSC/red-flag laps.
- **Leak-free scorecard:** `validate_scorecard.py` uses leave-one-race-out tyre fits and qualifying-based pace.
- **Predictor:** evaluated walk-forward. Each race is predicted using only earlier races and compared against uniform, grid-order, championship-order and recent-form baselines.

## Limitations

- **One car at a time.** The simulator models a single car in isolation, without traffic or overtaking. A full-grid multi-car simulation is the next planned step.
- **Data sources.** FastF1 live timing has required an F1 TV subscription since October 2026, so new races come from OpenF1 and Jolpica.
- **Local only.** The app is not deployed yet and runs locally.

## Deploying

Frontend on Vercel, API on Render, both free tier. Neither needs a secret; the order is Render first (to get the API URL), then Vercel, then one more env var on Render.

### 1. Backend on Render

1. Push this repo to GitHub (`karanir2005/F1-strategy-simulator`). `data/` must be committed: the API reads it at runtime.
2. render.com > sign in with GitHub > **New +** > **Blueprint**.
3. Pick the repo, branch `main`. Render reads `render.yaml` and shows one web service, `f1-strategy-api`. Click **Apply**. (It asks for `ALLOWED_ORIGINS`: type `http://localhost:3000` for now, you fix it in step 3.)
4. Wait for the first build (a few minutes). When the service shows **Live**, open `https://<service>.onrender.com/health`. It should return `{"status":"ok"}`. Copy the service URL.

Manual alternative to the Blueprint: **New +** > **Web Service** > pick the repo, Language **Python 3**, Root Directory empty (repo root), Build Command `pip install -r requirements-deploy.txt`, Start Command `cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT`, Instance Type **Free**, Health Check Path `/health`, env var `PYTHON_VERSION` = `3.12.8`.

### 2. Frontend on Vercel

1. vercel.com > sign in with GitHub > **Add New...** > **Project** > import the repo.
2. **Root Directory**: click Edit and choose `frontend`. Framework preset auto-detects as Next.js. No `vercel.json` is needed.
3. Under **Environment Variables** add `NEXT_PUBLIC_API_URL` = the Render URL from step 1 (https, no trailing slash, e.g. `https://f1-strategy-api.onrender.com`). It is baked in at build time, so changing it later needs a redeploy.
4. Click **Deploy**. Copy the production URL (e.g. `https://f1-strategy.vercel.app`).

### 3. Point the API at the frontend (CORS)

Render dashboard > your service > **Environment** > set `ALLOWED_ORIGINS` to the Vercel URL, e.g. `https://f1-strategy.vercel.app` (no trailing slash; comma-separate several, e.g. add a custom domain). Save; Render redeploys. Unset, it defaults to `http://localhost:3000,http://127.0.0.1:3000`. Vercel preview deployments have different URLs and will be blocked by CORS unless you add them.

### What to expect

- **Cold starts:** a free Render service sleeps after about 15 minutes without traffic, and the next request takes roughly 30-60 s to wake it (then the first `/predict/next` takes a few more seconds to load the tyre model). The frontend shows "cannot reach the backend" if you hit it mid-wake; retry. Free instances also get a limited monthly hour allowance.
- **Memory:** the 512 MB free limit is fine. Measured locally: about 140 MB after startup, about 185 MB after `/predict/*`, peak about 300 MB for a worst-case `/simulate` (50,000 sims, safety cars on).
- **Not available when deployed:** the in-app chat (`/chat`) needs a local Ollama and reports itself unavailable; the MCP server is local-only; nothing at runtime calls FastF1 or the `cache/` folder.

### Rolling new race data forward

The deployed API only has what is committed in the repo (`data/processed`, `data/staging`, `data/tracks`, `data/jolpica`, `data/openf1_predictor`). To add a race: run the pull/prediction scripts locally as usual, commit the changed files under `data/`, and push to `main`. Render redeploys automatically (`autoDeploy: true`) and the new data is live once the build finishes. There is no database and no runtime refresh. After each deploy the instance restarts cold, so the first request is slower.
