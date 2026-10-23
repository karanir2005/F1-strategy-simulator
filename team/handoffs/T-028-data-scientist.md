# T-028 — data-scientist — per-track tyre wear, calibration scripts

**Status:** done. pytest 314 passed + 3 new (`test_t028_ds.py`); 4 failures in `test_t012_qa.py` are `predict_race.py` `args.out_dir` (the other agent's in-flight edit, not mine). **Next:** qa-engineer; backend-engineer for the flag below.

## 1. Per-track wear (partial pooling) — shipped OPT-IN, global stays default
- `tire_model.fit_track_wear` / `model_for_track(model, track_id)`; `build_tire_model()` now also returns `model["_track_wear"][track_id][compound] = {age_coef, global_age_coef, n_stints, n_laps, pooled}` (16 tracks with a 2026 race). Per track: offsets and per-race fuel fixed at the global fit, driver-race fixed effects removed, bounded LS with SOFT>=MEDIUM>=HARD>=0 plus a ridge pull to the global slope (prior sd `TRACK_WEAR_TAU` = 0.01 s/lap/lap); compounds with < 3 stints at the track are pinned to global. Cliffs/max_reliable_age/offsets stay global.
- `data/tracks/track_params.json`: additive optional `tire_wear` per track (16/23), written by new `fit_track_wear_params.py` (re-run after `build_track_params.py`, which rewrites the file). `test_track_params` allows the optional key.
- `optimizer.optimize(..., track_id=None)`: applies `model_for_track`. `check_optimizer_plans.py --per-track` (-> `optimizer_plans_per_track.json`).
- **Evaluation caveat:** with one 2026 race per circuit, LORO leaves the held-out track without data, so per-track == global by construction. Used instead: split-half by driver inside the held-out race (`validate_scorecard.py --split-half / --per-track`; train on other 15 races + half the drivers, score the other half, two folds). Alternating-driver halves share the race, so this is a weaker test than LORO; it is the only one available.

| 16 races, quali_ratio | LORO global (T-020 shipped) | split-half global | split-half per-track tau .01 (chosen) | tau .02 | tau .04 |
|---|---|---|---|---|---|
| per-stint MAE excl SC (s/lap) | 1.554 | 1.545 | 1.486 | 1.459 | 1.445 |
| relative MAE | 0.730 | 0.723 | 0.710 | 0.711 | 0.712 |
| finishing-order rho (grid .834) | 0.870 | 0.874 | 0.876 | 0.873 | 0.872 |
| strategy-pair accuracy (n=277) | 0.661 | 0.664 | 0.661 | 0.650 | 0.639 |

Read: MAE and relative MAE improve a little (mostly via bias, -0.88 -> -0.86), rho +0.002, pair accuracy is -1 pair at tau .01 and -4 pairs at .02 (CI +-0.07: all within noise). Not strictly "no worse" on pair accuracy, and the split-half protocol is lenient, so **default is unchanged**; per-track is used only when a caller passes `track_id` / calls `model_for_track`. tau was picked from this same grid (small tuning leak, flagged). Per-track slopes vary a lot (Barcelona S/M .109, Madrid H .021, Austria .07, Monaco/Canada ~global), many S=M ties from the ordering bound.

| check_optimizer_plans (23 tracks) | global | per-track |
|---|---|---|
| top plans with SOFT | 18 | 17 |
| with 2 SOFT stints | 6 | 6 |
| stop count matches top-10 mode (17 comparable) | 17 | 17 |
| same compound set as winner | 5 | 5 |
| median / max gap to winner plan (s) | 2.3 / 18.9 | 2.3 / 15.8 |

Per-track barely changes plan shape (SOFT-heavy plans persist); the SOFT preference comes from the global offset (-0.12 s) and cap quality, not wear pooling.

## 2. Calibration scripts (16 races)
| constant | old | new | change | action |
|---|---|---|---|---|
| lap noise, MAD robust std (`calibrate_lap_noise.py`, 14,164 laps) | 0.8 (0.77 on 10 races) | 0.602 (raw std 0.992, now also printed) | -25% | **updated to 0.6** |
| pit loss median (`calibrate_pit_loss.py`, 294 stops, ex-Monaco) | 23.5 | 22.88 | -2.6% | left |
| pit loss noise std | 5.0 | 5.06 | +1% | left |

`calibrate_pit_loss.py` now reads `data/processed/*_pit_stops.csv` with the `build_track_params.pit_durations` filter (no FastF1). Stale "10 races" comments fixed in `monte_carlo.py` and `race_simulator.py`. Lap-noise 0.6 has no pinned test; `optimizer_plans.json` regenerated (unchanged headline numbers).

## Interface for backend-engineer (not done by me)
`api/routes/tracks.py` should call `optimize(..., track_id=t.track_id)` (and `TrackOut` may expose optional `tire_wear`) if per-track wear is to be switched on; suggest an `use_track_wear: bool = False` request flag. Default API behaviour is unchanged.
