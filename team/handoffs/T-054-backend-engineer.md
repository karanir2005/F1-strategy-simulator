# T-054 — backend-engineer — MCP server

**Status:** done, except the `simulate_grid` tool (blocked on T-034)
**Next owner:** orchestrator — commit; once `backend/app/core/race_grid_sim.py` lands, add `simulate_grid` (TODO marked at the bottom of `mcp_server.py`, before `TOOLS`) and a `forced_sc_lap` pass-through on `simulate_strategy`/`optimize_strategy`.

## What I did
- `backend/app/mcp_server.py`: stdio MCP server. Installed `mcp` **2.3.0** (Python 3.14 venv works). In 2.x `FastMCP` is gone; the class is `MCPServer` (`from mcp.server.mcpserver import MCPServer`). Tools: `list_tracks`, `predict_race(round=None, top_n=22)`, `get_race_results(round, season=2026)`, `optimize_strategy(track_id, max_stops, n_sims, top_n, total_laps, pit_loss_s, base_laptime_s, use_track_wear, safety_cars, seed)`, `simulate_strategy(strategies, track_id, base_laptime, n_simulations, pit_loss_seconds, safety_cars, seed, include_lap_times)`, `query_laps(race, driver, lap_from, lap_to, compound, green_only, columns, max_rows)` (hard cap 200 rows, `truncated` flag), `search_project_notes(query, k)`.
- In-process: optimize/simulate call the FastAPI route functions with the same pydantic request models (same validation and bounds as the API; HTTPException/ValidationError become `{"error": ..., "caveats": [...]}` results). Predict uses `prediction_service`; results use `jolpica_client.results` (offline cache first, network fallback).
- stdout safety: stray prints are redirected to stderr during each call; verified a real stdio subprocess round trip.
- RAG: `search_project_notes` chunks `team/research/*.md`, `team/handoffs/*.md`, `team/ROADMAP.md`, `README.md` by heading (fenced `#` ignored; 408 chunks) and ranks with a pure-Python BM25 (heading and file stem indexed too). Returns path, heading, score, snippet (~500 chars around first hit). Index is built lazily once per process (restart the server to pick up edited notes).
- Every result has `caveats` (single-car sim, mostly 1-stop plans, win_probability meaning, ~1 s gaps meaningless, SOFT/MEDIUM/HARD tie on pace and cliff placeholders, predictor model scope, wet-race lap data).
- `.mcp.json` at repo root (relative `.venv/Scripts/python.exe -m app.mcp_server`, cwd `backend`). README section added with Claude Desktop snippet.
- `team/research/T-054-mcp-eval.md`: 12 canned questions with expected tool and facts.

## Files changed
- backend/app/mcp_server.py (new), backend/tests/test_mcp_server.py (new, 12 tests incl. in-memory `mcp.Client` round trip), .mcp.json (new), requirements.txt (+`mcp==2.3.0`), README.md (MCP section), team/research/T-054-mcp-eval.md (new).

## Verification
- `pytest backend/tests/test_mcp_server.py`: 12 passed (~5 s). Full suite: 377 passed in 120 s.
- Manual: `predict_race()` live next race (Las Vegas R21, pre_quali) 6.7 s; stdio subprocess via `mcp.Client(StdioServerParameters(...))` listed 7 tools and returned results.
- Not verified: an actual Claude Desktop session (needs a restart on the user's machine). `predict_race()` with no round needs the Jolpica cache or network.

## Interface / contract changes
No HTTP API changes. Claude Desktop config (`%APPDATA%\Claude\claude_desktop_config.json`):
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
Claude Code: `.mcp.json` is picked up automatically (user approves on first use); or `claude mcp add f1-strategy-simulator -- .venv\Scripts\python.exe -m app.mcp_server` with cwd `backend`.

## Requests for other roles
- Whoever finishes T-034: `race_grid_sim.py` did not exist when I finished, so no `simulate_grid` tool. Then wire it in and add `forced_sc_lap`.

## Open questions / risks
- Tool results are plain dicts (no output schema), so clients get JSON text, not `structured_content`.
- Claude Desktop launches with its own cwd; `cwd` in config is required because the module is `app.mcp_server` under `backend/`.
- Brazil 2026 laps are wet (inters); `query_laps` caveats say so.
