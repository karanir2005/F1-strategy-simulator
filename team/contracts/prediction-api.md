# Contract: prediction API + two-page UI (T-022 to T-025)

User request (2026-10-19): redesign the UI to look polished (not average vibecoded), use framer-motion and other UI ideas from top-tier sites, and split into two pages: **Prediction** (race winner/podium) and **Strategy** (track picker + optimal stint plan + manual comparison).

## API (T-023, backend-engineer)
`GET /predict/next` -> latest prediction for the next race (US GP, round 18), computed with `backend/app/core/predictor.py` in the same way as `predict_race.py` (auto grid mode when qualifying exists, else pre-quali). Cache the result in memory keyed by (round, mode, qualifying-published) for 10 min; never call the network on each request beyond what predict_race already does; if Jolpica is unreachable fall back to the newest saved CSV in `data/jolpica/predictions/` and set `"stale": true`.
```json
{"race": {"round": 18, "name": "United States GP", "circuit": "Circuit of the Americas", "date": "2026-10-25", "quali_date": "2026-10-24"},
 "mode": "pre_quali" | "grid", "generated_at": "ISO", "stale": false,
 "drivers": [{"code":"ANT","name":"Kimi Antonelli","team":"Mercedes","grid":null,
              "p_win":0.263,"p_podium":0.57,"p_top10":0.9,"exp_pos":3.1,"p_dnf":0.08}],
 "headline": {"winner":"ANT","podium":["ANT","HAM","RUS"]},
 "evidence": {"walk_forward": {"races": 16, "win_logloss": 0.079, "baseline_win_logloss": 0.0937, "mode":"grid"}, "notes": ["..."]},
 "disclaimer": "..."}
```
Sorted by p_win desc. Probabilities sum to 1 (win), 3 (podium), 10 (top10). Also `GET /predict/history` -> per-race 2026 table from `data/jolpica/predictions/walk_forward_2026_by_race.csv` (round, race, grid winner, actual winner, p_actual_winner, predicted podium vs actual) for a "how it did" view.

## UI (T-024, frontend-engineer)
Two routes: `/predict` (default landing) and `/strategy`. Shared nav/layout. See `team/research/T-022-ui-inspiration.md` for the design direction.

## Addendum T-027: grid source and post-penalty override (backend-engineer)
Jolpica's qualifying endpoint is the classification, not the starting grid, so penalties are missing. The operator can supply the real grid:

- File: `data/jolpica/grid_override_2026_R18.csv` (pattern `grid_override_2026_R<NN>.csv`), columns `code,grid`; pit-lane start = `0`; 22 rows. If it exists it wins over the classification for BOTH `predict_race.py` and `GET /predict/next`, and it also switches auto mode to grid even when Jolpica has no qualifying. Delete the file to go back to the classification. An invalid file (duplicate codes or grid slots, unknown code) makes the live path fail, so the API serves the newest saved CSV with `stale: true`; check the CLI output for the error.
- New optional response field on `GET /predict/next`:
  `"grid_source": "qualifying_classification" | "override_file" | null`
  null when `mode` is `pre_quali`, or when a stale legacy CSV has no record of it. Existing clients that ignore it keep working. Suggested UI: when `mode == "grid"` and `grid_source == "qualifying_classification"` show "Grid = qualifying classification, penalties not applied"; when `override_file` show "Post-penalty grid supplied". `evidence.notes` carries the same sentence.
- `drivers[].grid` is 0 for a pit-lane starter (already in the contract), now reachable via the override file.
- Cache: a cached `pre_quali` payload is re-checked after 120 s (instead of 600 s) once `race.quali_date` is today or past, so the switch to grid mode appears within about 2 min. The cache key in the service now includes `grid_source`.
- CLI saved CSVs gain a `grid_source` column (filename unchanged: `2026_R18_<prequali|grid>_<YYYYMMDD_HHMM>.csv`). Note the CLI/CSV spell the mode `prequali`; the API spells it `pre_quali`.

## Addendum T-029: "what changed since pre-quali" fields (backend-engineer)
When `GET /predict/next` has `mode == "grid"`, each `drivers[]` entry gains two optional fields:
`"p_win_prequali": number | null`, `"p_podium_prequali": number | null` (probabilities in 0..1, same units as `p_win`/`p_podium`).
Source: the newest saved `data/jolpica/predictions/2026_R{round}_prequali_<YYYYMMDD_HHMM>.csv` for the same round (newest by filename stamp). They are null when no such file exists, the file is unreadable, or the driver is not in it. Nothing is estimated or back-filled. In `pre_quali` mode the fields are absent. Change per driver = `p_win - p_win_prequali`. Existing clients that ignore the fields keep working. The stale-CSV fallback attaches them too.

## Addendum T-035: qualifying (starting-grid) prediction (backend-engineer)
`GET /predict/next` gains an optional top-level `"qualifying"` block (null when unavailable; clients that ignore it keep working):
```json
"qualifying": {"available": true,
  "drivers": [{"code":"HAM","p_pole":0.16,"p_front_row":0.30,"p_q3":0.93,"p_q1_exit":0.02,"exp_grid":4.0,"actual_grid":null}],
  "evidence": {"races":16,"pole_logloss":0.1312,"baseline_pole_logloss":0.1371,"baseline_name":"championship order",
               "spearman":0.8493,"baseline_spearman":0.8268,"q3_logloss":0.2993,"baseline_q3_logloss":0.3637} | null}
```
- Probabilities are 0..1 from the same 20k-sim qualifying-order draw the race model uses. Sums over a 22-car field: pole 1, front row 2, Q3 10, Q1 exit 6. `p_front_row` = grid<=2, `p_q3` = grid<=10, `p_q1_exit` = slot >= 17 for 22 cars (16 advance from Q1; 20-car fields: slot >= 16). `exp_grid` = mean simulated qualifying rank (1 = pole). Drivers sorted by `p_pole` desc.
- These predict the qualifying classification, not the penalised grid.
- `pre_quali` mode: computed live (or from the stale CSV if it has the columns); `actual_grid` is null.
- `grid` mode: the numbers come from the newest `2026_R{round}_prequali_*.csv` that has the new columns (older files without them are skipped); `actual_grid` = the grid in use (classification or override file; 0 = pit-lane start), null for a driver not in the grid. If no such file exists the whole block is null.
- `evidence`: 2026 walk-forward vs the strongest non-uniform baseline by pole log-loss (here "championship order"; the other is "previous-race qualifying order"). null if `data/jolpica/eval/quali_walk_forward.csv` is missing. Produced by `python predict_race.py --evaluate-quali`.
- CLI prequali CSVs gain columns `p_pole,p_front_row,p_q3,p_q1_exit` (exp_grid already existed).
- Fix in the same change: `drivers[].p_win_prequali` / `p_podium_prequali` (T-029) were being dropped by the route's response model; they are now declared and the route uses `response_model_exclude_unset`, so they appear in grid mode and stay absent in pre_quali.

## T-038 additions (optional fields, existing clients unaffected)
- `stale_reason: string | null` on `/predict/next`: set only when `stale` is true because the live path raised (exception text, max 500 chars); logged server-side. After a failed refresh the stale payload is cached 60 s.
- `race.race_start_utc` / `race.quali_start_utc`: ISO UTC datetimes (e.g. `2026-10-25T19:00:00Z`) from the Jolpica schedule; null when the schedule has no time. Use these for countdowns instead of `date`/`quali_date`.
- Grid override file: `,` or `;` separator, case-insensitive headers (`code`, `grid`), codes trimmed/upper-cased, `PL`/`pit` = pit lane (0). Every entrant needs a row; unknown or missing codes and blank grid cells are errors (CLI message; API serves stale with `stale_reason`).

## Addendum T-044: race picker (past 2026 races)
No simulation at request time; both endpoints read `data/jolpica/predictions/walk_forward_2026_drivers.csv` (written by `python predict_race.py --by-race --offline`, one row per round x driver: grid-mode walk-forward prediction made from earlier rounds plus that race's own grid, plus the actual result). Re-run `--by-race` after each new race to add a round. File is re-read when its mtime changes.

- `GET /predict/races` -> `[{"round": 18, "name": "United States GP", "date": "2026-10-25", "has_prediction": true, "is_next": false}, ...]`, ascending by round: every scored round plus the next unraced one (`is_next: true`, served by `/predict/next`).
- `GET /predict/race/{round}` -> same `PredictionResponse` as `/predict/next` with `mode: "grid"`, `stale: false`, `grid_source: null`, `generated_at` = CSV mtime, `qualifying: null`, `p_win_prequali`/`p_podium_prequali` null, plus top-level `"race_done": true` and per driver
  `"actual": {"position": 1 | null, "classified": true | null, "status": "Finished" | "+1 Lap" | "Engine" | null, "dnf": false}` (`position` is the official classification slot, DNFs get a slot too; `dnf` true when not classified and status is not "Finished"/"+N Lap"). 404 if the round has no precomputed prediction.
- Frontend: `/predict?round=12` selects a past race; no param = next race.
