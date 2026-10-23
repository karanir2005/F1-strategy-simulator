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
