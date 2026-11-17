# T-059 QA report (milestone: T-034, T-053, T-054, T-055/T-061, T-056, T-057, T-058, T-060, old T-025 scope)

VERDICT: PASS WITH NOTES

No correctness bug in the simulation or policy search was found. Findings are robustness (loader), a validation-leak
label (pit policy), a misleading chat error, and unverified browser rendering (no browser tool).

## Per-item result

| # | Item | Result |
|---|------|--------|
| 1 | Diff review (grid sim, pit policy, chat, CORS) | PASS WITH NOTES (F1, F3, F5, F6) |
| 2 | Leakage | PASS WITH NOTES (F2 pit-policy validation not leave-one-out; F4 LOO is not walk-forward) |
| 3 | Headline numbers re-run | PASS (exact match) |
| 4 | pytest, tsc, lint | PASS |
| 5 | Browser / API pass | PASS WITH NOTES (no browser tool; /chat POST timed out, F3) |
| 6 | MCP stdio round trip | PASS |
| 7 | Deploy config | PASS |

## 4. Test runs
- `pytest` full suite from repo root: 426 passed (140 s), before my tests were added. With
  `backend/tests/test_t059_qa.py`: 12 pass, 3 xfail (strict; the 3 documented bugs below).
- `npx tsc --noEmit`: exit 0. `npm run lint`: no warnings or errors. `npm run build` not run, as instructed.
- Working tree has uncommitted edits by others during my run (predictor.py, team/BOARD.md, grid_sim_walkforward.py,
  data/predict_eval/t063_experiments.py); I did not touch them. The pytest and calibration_check runs used that tree.

## 3. Headline numbers re-run
- `validate_grid_sim.py --no-write` (n_sims 2000, selection table read from CSV, 239 s). All match T-060:
  calibrated sim Spearman 0.707 +/- 0.036, top-3 0.579, winner 0.684; t034_leaky 0.703 / 0.579 / 0.579;
  leak_fixed 0.700 / 0.596 / 0.579; fitted 0.710; PL 0.702; grid 0.657; passes 22.0 vs real 23.2; position-change sd
  3.99 vs 3.99; 2025: sim 0.655, grid 0.681, PL 0.657, passes 22.1/race. (data/scorecard/grid_sim_validation.json NOT overwritten.)
- `data/predict_eval/calibration_check.py`: matches T-057 (2026 Brier win 0.0212, podium 0.0650, top10 0.1519; log loss
  0.0758 / 0.2058 / 0.4718; ECE 0.0177 / 0.0309 / 0.034; temperature recalibration hurts held-out).
- 2025 run prints `2025 R6 miami skipped: ValueError('cannot convert float NaN to integer')`. T-060 handoff already
  discloses the Miami skip. It is a swallowed exception with the cause unfixed (note, F7).

## 1/2. Findings

F1 (low, robustness) - `backend/app/core/race_grid_sim.py:170-214` (`load_grid_calibration`) and `:246-303`
(`load_interaction_params`). Docstrings say "Never raises for a missing or malformed file". Not true for well-formed JSON with
bad content: `{"sc":{"gap_s":"abc"}}` raises ValueError (cast at line 197); a file containing `[]` raises AttributeError
(line 184, `d.get`); `{"overtake": null}` raises AttributeError in `load_interaction_params` (line 276). Impact: a hand-edited
or truncated-schema JSON takes down every /simulate/grid and /strategy/policy request instead of falling back to placeholders.
Tests: `test_t059_qa.py::test_calibration_non_numeric_value_does_not_raise`, `test_calibration_json_list_does_not_raise`,
`test_interaction_null_section_does_not_raise` (xfail strict; remove the marker once fixed). Owner: backend-engineer.

F2 (medium for honesty, low for numbers) - `validate_pit_policy.py:67-70` + `PP.build_context`. The module docstring calls it a
validation, and the script `get_tire_model()` / `get_track()` are built from all 2026 races including the race being checked
(tyre model, base lap, pit loss, stint caps, and the grid itself). The policy never sees the lap data's pit/SC rows,
so the compared quantity (box or not on real SC laps) is not directly leaked, but tyre ages / caps / the chosen compound do come
from a model that saw that race. Not leave-one-race-out, unlike validate_grid_sim.py. Expected effect: small. Fix
options: label it "in-sample for the tyre model/track entry", or reuse `calibrate_grid_sim.Study` fold contexts. Owner: data-scientist.

F3 (medium, UX/correctness of message) - `backend/app/services/chat_service.py:156-160`. A `requests` ReadTimeout (Ollama slow,
CHAT_TIMEOUT_S 180) is reported as `503 "Cannot reach Ollama ... Install Ollama and run ollama pull"`. Ollama was
running, the model was pulled, and `/chat/status` said available. Repro: POST /chat on :8001 with
`{"messages":[{"role":"user","content":"List the tracks you know about, briefly."}],"context":{"track_id":"brazil"}}`:
two attempts both returned that 503 after ~188 s; `/api/ps` shows size_vram 0 (CPU-only) and context_length 4096. (A
separate TestClient chat earlier in the session, which does call real Ollama, did return 200 with a correct list_tracks answer, so
the loop works but is slow on this machine.) Suggest: catch `requests.Timeout` separately ("model is still thinking, try a smaller question
or raise CHAT_TIMEOUT_S") and keep the "install" text for ConnectionError only. Also note: an abandoned HTTP request does not cancel the
Ollama generation, so the second request queued behind the first, which likely made the repeated timeout worse. Owner: backend-engineer.

F4 (low, label) - `validate_grid_sim.py` / `calibrate_grid_sim.py` are leave-one-race-out, not walk-forward: constants
(pace scale, noise, SC, strategy mix, tyre fit) for race R are fitted on all other 2026 races including LATER ones. The docstring
says "leave-one-race-out" so this is consistent with the label, but it is not the "no future data" standard the project uses for
the ML predictor. Held-out race results are used only for scoring (confirmed in `Study.score`); the overtake (scale, shift) choice
excludes R from its targets (`choose_config(exclude_round=rnd)`). Residual leaks are listed honestly in the T-060 handoff
(selection table built with all-race constants, dirty-air / pass-cost scalars in race_interaction.json not refit per fold, shared strategy-mix
stats). I accept them as small. The reported validation numbers do not include the held-out race beyond those disclosed items. Owner: none, note only.

F5 (info) - Docstring typo, `race_grid_sim.py:20-22`: "a car whose free lap would put it whose gap to the car in front ...".
Garbled sentence in the module docstring step 4. Owner: backend-engineer.

F6 (info) - pit rejoin: a car that stops on a green lap is excluded from the catch/overtake line (`alive = ~dead & ~pit`,
line 583) and gets no minimum spacing against the cars it lands among. Two cars can end the lap with nearly equal clock
times; the next lap's sort and dirty-air rule handle it. No wrong ordering was produced in my invariant tests (all runs yield permutations,
finish times are monotone in position, retired cars always last, DNF draws unchanged when another car's plan changes), so I
rate it a simplification not a bug.

F7 (info) - `POST /simulate/grid` with a too-short override returns `"car 0: strategy covers 10 laps, race is 71"`: car index not driver code
(`_strategy_tables` message). Owner: backend-engineer, cosmetic.

F8 (info) - Setting `ALLOWED_ORIGINS=""` (set but empty) yields an empty origin list (all cross-origin calls blocked) instead of the
local default. Parsing otherwise correct: whitespace, trailing slash and empty items handled (test `test_cors_env_parsing`).
Preflight from http://localhost:3000 returns allow-origin; an unlisted origin gets 400. Owner: backend-engineer, only matters if Render env is set blank.

## Claims checked and confirmed
- Window-search exactness (pit_policy.py:20-29): reasoned independently and confirmed. Eligibility (tyre age >= k, reaction allowed) does not
  depend on the window or on earlier events, the policy fires once, so a windowed policy equals "fire at the first eligible start in the window", which is
  one of the slot tables as long as fewer than MAX_SLOTS=3 eligible events precede it (documented; slot s empty implies s+1 empty, so the `break` is right).
  Existing test `test_slot_tables_reproduce_the_policy_exactly` passes. Selection seed != reporting seed (seed+1), paired with the same base plan.
- `plan_hook=None` is bit-identical (existing test). Overtake bookkeeping (tail / m1 / m2) re-read: success leaves the defender as the tail and the attacker second, failure
  makes the attacker the tail; consistent. DNF cars sort last in order of earlier retirement.
- chat tool loop: capped at MAX_TOOL_ROUNDS then one tool-less call; garbage args, unknown tools, `{}` tool call, empty final content
  all handled (my tests). Tool exceptions become `{"error"}` dicts. Unknown args dropped.
- Deploy: `app.main` imports cleanly with fastf1, mcp, matplotlib, sklearn, xgboost blocked via an import hook (and /health, /chat/status work). `mcp`
  is imported only inside `mcp_server.main()`; `fastf1` only in `data_loader.py`, which only `openf1_loader.py` imports, and no app module imports that.
  Runtime third-party imports under backend/app = fastapi, uvicorn, pydantic, numpy, pandas, scipy, requests; all in requirements-deploy.txt with pins
  identical to requirements.txt. render.yaml build/start/health path fine; the data files the API reads (data/sim_params, data/tracks, data/jolpica, data/processed) are git-tracked.
  Not verified by me: an actual Render build (T-056 says verified on Python 3.12).

## 5. Browser / API pass (no browser tool: curl against live servers)
- Frontend :3000: /predict 200, /strategy 200, no server error text. Page content is client-rendered, so I could not check on-screen rendering or
  console errors. `app/strategy/page.js` bundle contains the calls `/strategy/policy`, `/chat`, `/chat/status` and the `if_field_follows` field; it has no `/simulate/grid`
  call (the grid sim has no UI, only API/MCP/chat). Visual pass still to be done by a human or a browser-capable agent.
- API :8001: /health ok; /chat/status available (qwen2.5:7b pulled); /predict/races, /next, /race/20, /history all 200;
  POST /simulate/grid brazil round 20 n_sims 500: 200 in 0.47 s (NOR grid 1, p_win 0.38, 71 laps, grid source "race grid"); n_sims=1 with forced_sc_lap=30: 200;
  POST /strategy/policy brazil round 20: 200 in 19 s, rule fires for NOR "SC between laps 2-64, tyres >= 8 laps, box Mediums", significant true,
  gain 0.394 +/- 0.078 pts, if_field_follows 0.60 +/- 0.15 (larger than the solo gain, worth the DS double-checking against the docstring's "nobody gains from a free stop that everybody takes" reasoning); errors: unknown track 404, bad stint total 422, empty chat messages 422.
- Pydantic `PolicyResponse` / `PolicyOut` / `GainOut` match `frontend/lib/api.ts` field for field (frontend `PolicyRequest` omits `round`; optional).
- POST /chat: see F3, not confirmed end to end through uvicorn in the 180 s limit.

## 6. MCP stdio round trip
Official `mcp` client over stdio against `python -m app.mcp_server` with `PYTHONPATH=backend`: initialize ok, 9 tools listed
(get_race_results, list_tracks, optimize_pit_policy, optimize_strategy, predict_race, query_laps, search_project_notes,
simulate_grid, simulate_strategy); list_tracks, simulate_grid (brazil r20, 200 sims) and search_project_notes (489 chunks) succeed; a bad track id returns an
`{"error": ..., "caveats": [...]}` payload with is_error False (soft error by design, same as chat).

## Tests added
`backend/tests/test_t059_qa.py`: edge cases (n_sims=1, 1-lap and 2-lap races, single car rejected), invariants (permutations, retired last,
finish-time monotone), determinism and CRN of DNFs, a pit-stop-cost check, chat loop with garbage tool calls and the round cap, CORS env parsing,
and the 3 strict-xfail tests for F1.
