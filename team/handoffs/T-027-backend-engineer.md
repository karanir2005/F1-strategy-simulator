# T-027 — backend-engineer — grid-mode rehearsal, grid override, practice pace

**Status:** done. Full pytest after the last fix: 314 passed, plus the 4 test_t012_qa tests that failed mid-way and now pass (10/10 rerun). 9 new tests in `backend/tests/test_t027_grid_mode.py` (no network, no writes to data/jolpica).
**Next owner:** qa-engineer; frontend-engineer for the `grid_source` note.

## 1. Rehearsal (fake 22-driver grid, RUS = 0, `--offline --round 18 --out-dir <scratch>`)
CLI works: grid mode, pit-lane starter handled, sums 1/3/10, saved as `2026_R18_grid_<stamp>.csv`. Bugs and gaps fixed:
- Rehearsing would have written a fake grid CSV into `data/jolpica/predictions/`, which the API's stale fallback would then serve. Added `--out-dir`.
- The CLI and API decided mode/grid in separate code. Both now call `predict_race.resolve()`.
- The CLI printed no headline. It now prints `headline: winner X, podium A B C` (same rule as the API, tested equal).
- A cached pre_quali payload could linger 10 min after qualifying went live. From quali day on it is re-checked after 120 s.
- Verified by test: pre_quali without qualifying, switch to grid when R18 quali appears (fabricated from R17 in a mock), CSV naming, stale fallback reads the CLI CSV, mode/headline agree across CLI and API. Evidence is API-only (the CLI does not print it). Mode spelling differs by design: CLI/CSV `prequali`, API `pre_quali`.

## 2. Grid penalties
- Override file `data/jolpica/grid_override_2026_R18.csv` (`code,grid`; pit lane 0). If present it wins for CLI and API, and also forces grid mode. A bad file raises in the loader, so the API shows the stale CSV; check by running the CLI.
- New optional field on `/predict/next`: `grid_source: "qualifying_classification" | "override_file" | null`. Also stored as a CSV column so stale responses keep it. Backward compatible. Full text in `team/contracts/prediction-api.md` (Addendum T-027).

## 3. Practice pace (--fp), 2026 walk-forward R2-17, same hyperparameters, 10k sims
| mode | FP | win logloss | podium logloss | winner NLL |
|---|---|---|---|---|
| pre-quali | off | 0.1133 | 0.2456 | 1.641 |
| pre-quali | on | 0.1139 | 0.2474 | 1.646 |
| grid | off | 0.0790 | 0.2075 | 1.103 |
| grid | on | 0.0798 | 0.2094 | 1.121 |

On is worse on both metrics in both modes. Per race, FP improves win log-loss in 10/16 rounds in each mode but podium in only 8/16 (pre-quali) and 6/16 (grid); the gains are tiny and R2 (+0.010 win, +0.022 podium in grid) and R10 (+0.033 win) lose far more. That fits the caveat: teams run different fuel and engine modes in practice. **Recommendation: leave `use_fp` off.** The decision rule is not met.

## What the user must run on Saturday (after qualifying is final, from repo root)
1. `.venv\Scripts\python.exe predict_race.py --mode grid` (needs network; Jolpica may lag the session by a while. If it exits "no qualifying", wait or go to step 2).
2. If any grid penalty was announced: write `data/jolpica/grid_override_2026_R18.csv` (`code,grid`, all 22 rows, pit-lane starter = 0) and run `.venv\Scripts\python.exe predict_race.py --mode grid --grid-file data\jolpica\grid_override_2026_R18.csv`. The API picks the same file up on its own; no flag needed.
3. Check the CLI line `grid_source:` and `headline:`. The web page (`GET /predict/next`) refreshes within about 2 min (600 s once in grid mode); restart uvicorn if you want it immediately.
4. Do not pass `--fp`.

Not touched: cache/, frontend/, tire_model.py, optimizer.py, race_simulator.py. Scratch: fp experiment in the scratchpad (`fp_eval.py`, `fp_out.txt`). `walk_forward_2026.csv` was not regenerated.
