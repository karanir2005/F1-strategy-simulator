# T-002 — data-scientist — Stint detection fix + richer lap extraction + re-pull

**Status:** done
**Next owner:** qa-engineer: verify T-002 (see "For QA" below). After that, data-scientist picks up T-003 (scorecard), using the notes at the end.

## What I did
- **Rewrote `data_loader.extract_laps_df`** following the T-001 brief (§2, §4).
  - Stints come from FastF1 `Stint`, split into real tyre changes and same-tyre pit-lane passes (`tyre_stint`, `stint_id`, `stint_start_age`, `is_tyre_change`, `is_pit_pass_no_change`, plus the raw `ff1_stint`).
  - Rows with a NaN `LapTime` are now kept. `FastF1Generated` rows and DNS drivers are dropped.
  - `TrackStatus` is decoded per character. Added gaps (race order and on-road), pit columns, `overtake_enabled`, weather at lap end, and `compound_c`.
  - Hungary PER laps 22–46: the whole FastF1 stint has no tyre data, so forward-filling the compound would invent one. These laps are flagged `tyre_data_missing` and left NaN. Partial gaps inside a stint are imputed (`tyre_data_imputed`; none occur in the current data).
- **Sidecars** per race in `data/processed/`: `{gp}_2026_results.csv`, `_weather.csv`, `_track_status.csv`, `_rcm.csv`, `_pit_stops.csv`, and `_race_info.json`.
  - `_pit_stops.csv` has one row per pit-lane passage, with tyre-change, SC/VSC/red and compound before/after flags.
  - `_race_info.json` holds `compound_map`, its source, `total_laps`, `event_format` and `event_date`.
  - **The C-compound mapping** lives in `data_loader.COMPOUND_ALLOCATION`, keyed by `(year, EventName)`. It is written per lap as `compound_c` and per race in `race_info.json`.
- **`pull_races.py`**
  - Runs in FastF1 offline mode by default, so every re-extract is guaranteed to be a cache hit. `--online` allows network access and `--out` sets the output directory.
  - **Spain is removed from RACES** with a note pointing to T-009. Left in, it would have triggered a network pull. I removed it from `calibrate_pit_loss.py` for the same reason.
  - Re-extracted all 10 races. A second run produced byte-identical laps CSVs.
- **`tire_model`**
  - `assign_stint_ids` uses `tyre_stint`. Legacy CSVs fall back to the old rule.
  - `clean_laps` now also drops lap 1, out-laps on **any** set, restart laps (the first lap after SC/VSC/red), NaN lap times and unknown-tyre laps.
  - **Local-yellow-only laps: I deliberately kept excluding them.** Against each driver's neighbouring green laps they are +0.09 s median, but +0.84 s mean and +1.56 s p90 (green laps: −0.00 / +0.10 / +0.82). That is a heavy incident tail for only about 4% more laps.
- **N-3 root cause and fix: the estimator, not the data.**
  - The old fit regressed each compound on `[1, lap, age, age²]` with its own fuel slope (−0.026 to −0.038), then discarded that slope. The simulator used a different one (−0.054). So the wear curves were fitted against a fuel term the simulator never applies.
  - SOFT's quadratic took the whole fuel trend (age_coef −0.055 ≈ −fuel), which the monotone clamp then turned flat.
  - Cleaning alone does not fix it. The old estimator on the new data is still flat until about age 28.
  - New `fit_joint_model`: one within-driver-race fixed-effects regression with compound offsets, a lap_number slope per race (fuel plus that track's evolution) and per-compound wear. The simulator's `_fuel_coef` is the lap-weighted mean of the per-race slopes, so wear and fuel are now consistent.
  - **Linear wear is the default** because it beats quadratic in leave-one-race-out (LORO) validation. `build_tire_model(quadratic=True)` restores the quadratic. Full reasoning is in the `tire_model.py` module docstring, dated 2026-10-19.
- **`validate_replay`**
  - `reconstruct_strategy` uses `tyre_stint` and sets `Stint.start_age = stint_start_age` (the T-004 request).
  - The actual total is now measured at the timing line, because NaN-LapTime laps are kept.
  - Each replay also prints its error excluding SC/VSC/red laps.
  - Added two replays that include used-set stops: Britain PIA and Hungary VER.
- `calibrate_lap_noise.py` now uses `clean_laps`, so it is NaN-safe. I also fixed a latent `fit_compound_pace_offsets` crash on float `tire_age` (`predict_delta` received a float).

## Files changed
- `backend/app/core/data_loader.py`: extraction rewrite, sidecar extractors, `pit_stops_from_laps`, `get_race_tables` / `save_race_tables`. `get_race_data` is kept as a compatible wrapper.
- `backend/app/core/tire_model.py`: T-002 stints and cleaning, `fit_joint_model`, new `build_tire_model` (adds `quadratic` and `laps` params), docstring with measurements. Legacy fitters are kept.
- `validate_replay.py`: `start_age`, timing-line totals, SC-excluded error, 2 used-set replays.
- `pull_races.py`: offline by default, sidecars written, Spain removed (T-009).
- `calibrate_lap_noise.py`: uses `clean_laps`. `calibrate_pit_loss.py`: Spain removed.
- `data/processed/*`: 10 laps CSVs re-extracted (11,313 → 11,473 rows, 8 → 49 columns) plus 60 sidecar files.
- Tests: `backend/tests/test_data_loader_extraction.py` (new, 13 tests), `backend/tests/test_validate_replay.py` (new, 5), `backend/tests/test_tire_model.py` (+9).

## Verification
- **pytest (repo root): 166 passed.** That includes the concurrent backend and QA tests. Mine are the 27 new tests listed above.
- **Merged-stint errors: 58 → 0.** These are real tyre changes the old `tire_age == 1` rule fails to split: 57 stops onto used sets (start age 2–9) plus 1 stop onto an unknown set (Hungary PER).
  - On the new CSVs the old rule gives 35 stints containing more than one compound and 56 within-stint age jumps. The new stints give 0 and 0, which is enforced by `test_real_data_has_no_merged_stints`.
  - Counts match T-001 exactly: 343 tyre changes, 57 onto used sets, 44 same-tyre pit passes, and lap-1 used-set starts of China 2, Canada 7, Hungary 1.
  - Gaps: 0 negative, 0 missing for P>1.
  - Second, separate failure mode found: the old loader dropped NaN-LapTime rows, which sometimes deleted the age-1 out-lap. China GAS replayed as a **single 54-lap MEDIUM stint**. The truth is MEDIUM 10 + HARD 46 over 56 laps, with the stop under SC.
- **Degradation, s lost vs. age 2 at ages 5 / 10 / 20 / 30** (before = HEAD code on old data; after = new code on new data):

  | | before coefs | before deltas | after coefs (linear) | after deltas |
  |---|---|---|---|---|
  | SOFT | age −0.0547, age² +0.00145 | 0.00 / 0.00 / 0.00 / 0.00 | age +0.0105 | 0.03 / 0.08 / 0.19 / 0.29 |
  | MEDIUM | +0.0079, +0.00060, cliff 39 | 0.04 / 0.12 / 0.38 / 0.76 | +0.0376, cliff 38 | 0.11 / 0.30 / 0.68 / 1.05 |
  | HARD | +0.0299, −0.00032 | 0.08 / 0.21 / 0.41 / 0.55 | +0.0503 | 0.15 / 0.40 / 0.91 / 1.41 |

  - The fuel coefficient goes from −0.0539 to −0.0485 s/lap. Per race it ranges from −0.026 (Monaco) to −0.077 (China).
  - `max_reliable_age` is unchanged: S 37, M 51, H 48.
  - The SOFT curve now increases with age, which closes N-3. A used SOFT set is now slower than a new one.
- **LORO, within-stint lap-time shape of every ≥6-lap clean stint in the held-out race (full pipeline)**
  - RMSE: **1.311 s before → 1.157 s after** (quadratic variant: 1.163 s).
  - Slope bias (predicted − actual): MEDIUM −0.024 → −0.003 and HARD −0.038 → −0.003 s/lap. The old model under-predicted degradation.
  - Hungary per-race RMSE: 1.80 → 0.71.
  - The LORO harness is at C:\Users\rushi\AppData\Local\Temp\claude\C--Users-rushi-Desktop-Folders-Code-F1-strategy-simulator\bbb5655a-bfbb-4671-b4b7-bfaa13ddf821\scratchpad\loro.py
- **Compound pace offsets vs HARD (still inverted, placeholder still in use)**
  - Paired estimator: before S +0.35 / M −0.13; after S +0.32 / M +0.16.
  - New joint fit: S +0.38 / M +0.18.
  - Keyed by C-number instead (vs C3): C1 −0.47, C2 −0.28, C4 +0.18, C5 +0.25, so the inversion survives that too.
  - It also survives every cleaning step, so it is not a data artefact. This is a Phase 3 item, recorded in the docstring.
- **`validate_replay.py`**

  | Replay | Before (stint, laps) | Before error | After (stint, laps, start age) | After error | After, excl. SC/VSC/red laps |
  |---|---|---|---|---|---|
  | China GAS | MEDIUM 54 (wrong strategy, 2 SC laps missing) | +26.7 s (+0.50%) | MEDIUM 10, HARD 46 | −103.7 s (−1.83%) | +45.3 s (+0.89%) |
  | Britain GAS | 51 laps | −150.2 s (−2.96%) | 52 laps | −210.3 s (−4.02%) | +1.9 s (+0.05%) |
  | Austria GAS | — | −40.3 s (−0.77%) | — | −41.3 s (−0.79%) | +0.8 s (+0.02%) |
  | Britain PIA (new) | MEDIUM 2, HARD 49 (2 used-set stops merged) | −220.5 s | M2, H34, M11 (age 4), S5 (age 4) | −244.4 s (−4.67%) | −42.6 s (−1.03%) |
  | Hungary VER (new) | MEDIUM 14, HARD 56 (SOFT stint merged) | +13.5 s | M14, H27, S29 (age 4) | −0.0 s | +24.4 s (+0.42%) |

  - Headline % errors look worse because the actual totals now include the SC laps the engine doesn't simulate. Excluding those laps, the error is 0.02–1.03%.
  - Isolating `start_age` alone moves PIA by 1.4 s and VER by 0.9 s.
- **Other numbers**
  - Lap noise (`calibrate_lap_noise.py`): **1.03 s** with `clean_laps`. The HEAD script gives 2.76 s on the old data and 2.80 s on the new, so neither reproduces the 0.8 s in `monte_carlo` (see Requests).
  - Pit stops: green-flag real tyre changes have a median pit-lane time of **23.47 s** (n=198), which confirms `DEFAULT_PIT_LOSS_SECONDS = 23.5`. Under SC the median is 26.97 s (n=63), under VSC 23.16 s (n=66).
  - Stints ending past their compound's boundary (`max_reliable_age` / cliff): 18 of 546 overall, 1 of 67 used-set stints. This answers T-004's request.
- **Not verified**
  - `overtake_enabled` against per-lap ground truth. It is approximate: race-control laps are the leader's laps.
  - The Miami C-mapping, which comes from a secondary source.
  - The API/frontend with the new model end-to-end. The interface is unchanged, but `/simulate` numbers will shift with the new curves.

## Interface / contract changes
- **Laps CSV**
  - All old columns are kept with the same meaning. The exception is that `lap_time_seconds` is now **nullable**: about 160 rows, mostly SC/red laps.
  - New columns are documented in the `extract_laps_df` docstring (T-001 §4). I omitted `ff1_generated` because those rows are dropped, so it would always be False.
  - `stint_id` in the CSV is the string `"<driver>_<n>"`. Inside `tire_model`, `assign_stint_ids` overwrites it with the integer `tyre_stint`.
- **Tire model dict**
  - Same keys the engine reads.
  - `age2_coef` is now 0.0 by default (linear wear).
  - `_fuel_coef` now means "fuel + mean track evolution", fitted jointly with wear.
  - New diagnostic keys the engine ignores: `_fuel_coef_by_race`, `_compound_offsets_fitted`, `_n_laps_fitted`.
  - `build_tire_model(min_samples_per_age=3, quadratic=False, laps=None)`. Existing calls are unchanged.
- **New public functions**
  - `tire_model.fit_joint_model(clean, quadratic=False)`
  - `data_loader.get_race_tables`, `save_race_tables`, `pit_stops_from_laps`, `extract_results_df` / `_weather_df` / `_track_status_df` / `_rcm_df`, `compound_map_for_event`
  - `validate_replay.actual_race_time`
- No changes to `race_simulator`, `monte_carlo` or `api`.

## Requests for other roles
- **backend-engineer (optional, low priority)**
  - The `Stint` docstring says "About 29% of real post-pit stints … start on a used set". The correct figure is 57 of 343 real tyre changes (17%). The 29% lumped in same-tyre pit passes.
  - `DEFAULT_LAP_TIME_NOISE_STD` (0.8) isn't reproducible from `calibrate_lap_noise.py`, which now gives 1.03 s. Under CRN with iid noise this doesn't change rankings, so it can wait for T-003.
- **orchestrator**
  - `CLAUDE.md` still says "0.50–2.96% across 3 test cases" and "Spain" (pull_races). Suggested replacement: "5 replays; 0.02–1.03% excluding SC/VSC/red laps (the engine has no SC yet), 0.0–4.7% including them; LORO within-stint RMSE 1.16 s". I didn't edit it because it's not in my lane.
  - T-009 is still blocked on the Barcelona vs Madrid decision.
- **For QA (T-005)**, please verify:
  - 0 merged stints (`test_real_data_has_no_merged_stints`)
  - the used-set and same-tyre-pass unit tests
  - re-running `pull_races.py` is offline and byte-identical
  - the LORO before/after numbers (harness path above)
  - that `validate_replay` start ages match the laps CSV
  - that the SOFT curve is monotone increasing

## Open questions / risks
- **The SOFT curve is weak and form-dependent.**
  - The linear fit is +0.0105 s/lap. The quadratic fit is still U-shaped: SOFT gets faster until about age 13 and is flat until about 25.
  - LORO still says held-out SOFT stints improve faster than predicted (slope bias +0.11 s/lap, was +0.17).
  - Per-race SOFT wear is +0.04 to +0.08 at Australia, Austria, Belgium and Hungary, but about 0 or negative at Monaco (+0.015, post-red-flag sprints in a train), Canada (−0.023, where MEDIUM is negative too, which points to track evolution) and Miami (−0.012, 49 laps).
  - Canada and Monaco together hold about half of all SOFT laps, and SOFT is C5 at both.
  - A proper fix needs per-track terms (Phase 3). I tried median-of-races pooling and a clean-air filter (gap ≥1/1.5/2 s); both were worse or no better.
- **The model now degrades more.** MEDIUM/HARD at age 30 is about 2.5× the old value. Strategy comparisons in the UI will shift toward more stops. The change is justified by LORO, but it is user-visible.
- **Monaco red flag:** a few drivers (e.g. ALB) have two consecutive "tyre changes" on laps 68 and 69 onto SOFT age 7 → age 7. FastF1's TyreLife didn't advance, so the T-001 rule counts the second one as a change. This is harmless for fitting, because those laps are red/out-laps, but a Monaco replay would show an extra stop.
- **Hungary PER stint 2** (25 laps) has unknown tyres. It is excluded from fitting, and replaying PER raises a clear `ValueError`.
- **Notes for T-003:**
  - Score finishing order against `results.csv` `position`.
  - Use `build_tire_model(laps=…)` for leave-one-race-out; my scratchpad `loro.py` is a starting point.
  - `base_laptime` still leaks the race's own fastest lap.
  - Report errors with and without `is_neutralised` laps, because SC laps dominate the totals.
  - For strategy-ranking pairs, `_pit_stops.csv` has `is_tyre_change` and `pit_under_*` to pick comparable stops.
