# T-056 — backend-engineer — Deployment prep (Render + Vercel)

**Status:** done (nothing deployed; the Docker/Render build itself could not be run here)
**Next owner:** orchestrator/user — do the Render and Vercel click-through in README "Deploying"; QA optional.

## What I did
- Chose a native Python Render service (`render.yaml`), not Docker. Reasons: no Docker is installed here, so a Dockerfile could not be built or tested; the repo root has to be the build context either way (API reads `../data/`); native needs nothing else to maintain. Consequently no `.dockerignore` (not used). `cache/` is never read at runtime: no API route imports fastf1 or touches `cache/` (verified: `fastf1` not in `sys.modules` after importing the app and hitting all routes). Tracked `cache/` files do get cloned but are inert.
- Python version: the local venv is 3.14, but I did not rely on Render supporting 3.14. Pinned `PYTHON_VERSION=3.12.8`. Verified with `pip download --python-version 3.12 --platform manylinux --only-binary=:all:` that numpy 2.4.3, pandas 2.3.3, scipy 1.17.1 (and the pure-Python pins) have Linux cp312 wheels. Not verified: an actual install on Linux 3.12 (only 3.14 is installed on this machine; the 3.11 install dir is broken).
- `requirements-deploy.txt`: runtime subset (fastapi, uvicorn, pydantic, numpy, pandas, scipy, requests) at the same pins. Omits fastf1 (heavy, unused by the API) and mcp (local only). Must be kept in step with `requirements.txt` on pin bumps.
- CORS from `ALLOWED_ORIGINS` (comma-separated, whitespace/trailing slashes stripped), default `http://localhost:3000,http://127.0.0.1:3000`. Tested: allowed origin gets the header, other origin gets 400.
- `/health` already existed. Start command is in `render.yaml` (`cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT`), health check path set.
- Runtime data: Render builds from the git clone, so the data dirs the API reads (`data/processed`, `data/staging`, `data/tracks`, `data/jolpica/{raw,derived,predictions,eval}`, `data/openf1_predictor`) are all git-tracked (checked with `git ls-files`) and arrive automatically. `data/openf1/raw` (197 MB) is gitignored and not needed. Runtime has no network calls and no file writes in prediction_service/API. `data/sim_params/` does not exist yet (another engineer's work): if /simulate/grid reads it, it must be committed. I did not check grid_sim.
- Memory (Windows, Python 3.14, TestClient in-process; Linux should be similar): 15 MB bare, 141 MB after importing the app, about 185 MB peak after all `/predict/*`, 199 MB after /simulate 10k, 298 MB peak after /simulate 50k with safety cars (the request max). /optimize adds nothing measurable (it responds in ~0.03 s). Fits 512 MB. Did not measure /simulate/grid (route landed mid-task).
- Frontend: `frontend/lib/api.ts` already uses `NEXT_PUBLIC_API_URL ?? "http://localhost:8000"`; unchanged. No `vercel.json` needed (set Root Directory = `frontend` in the Vercel UI). `npx tsc --noEmit` clean; `npm run lint` clean; production build succeeded in a scratch copy with node_modules junctioned (dev server's `.next-dev` untouched) and `NEXT_PUBLIC_API_URL` set.
- README "Deploying" section: Render (blueprint and manual), Vercel, CORS step, cold start, memory, what is unavailable, rolling data forward (commit `data/` + push; autoDeploy redeploys).

## Files changed
- `C:\Users\rushi\Desktop\Folders\Code\F1-strategy-simulator\render.yaml` — new blueprint
- `C:\Users\rushi\Desktop\Folders\Code\F1-strategy-simulator\requirements-deploy.txt` — new runtime pins
- `C:\Users\rushi\Desktop\Folders\Code\F1-strategy-simulator\backend\app\main.py` — `import os`, CORS origins from `ALLOWED_ORIGINS`
- `C:\Users\rushi\Desktop\Folders\Code\F1-strategy-simulator\README.md` — "Deploying" section appended at the end (after Limitations)

## Verification
- `pytest` from repo root: 390 passed (141 s) with all concurrent work present.
- CORS preflight check with env var set (see above).
- tsc, lint, scratch production build: pass.
- NOT verified: the real Render build/boot, install on Linux 3.12, Render's `PYTHON_VERSION` handling, `region: oregon` choice, cold-start timings (the 30-60 s figure is Render's general behaviour, not measured).

## Interface / contract changes
- New env var `ALLOWED_ORIGINS` (backend). New env var `NEXT_PUBLIC_API_URL` (frontend, already supported). No API schema changes.

## Requests for other roles
- Whoever owns /simulate/grid and data/sim_params: any file read at runtime must be git-tracked and not under a gitignored path, and the route should not pull in fastf1/mcp. Re-measure memory with a full 20-car grid at max sims; it is the most likely thing to exceed 512 MB.
- Frontend: the /chat panel will show Ollama unavailable on the deployed site; hide or label it if desired. Optionally show a "waking the server" message when the first request fails (free-tier cold start).

## Open questions / risks
- `.gitignore` ignores `*.log`/`cache/`, fine; but `requirements-deploy.txt` and `render.yaml` are untracked until the orchestrator commits them. Render needs them pushed.
- Repo clone size on Render includes tracked `data/staging` (15 MB) and tracked `cache/` bits; fine.
- Free Render instances have monthly hour limits and public CORS-open-to-your-origin only; there is no auth or rate limiting on /simulate (50k sims per request), so a public URL can be hammered. Acceptable for a portfolio deploy.
