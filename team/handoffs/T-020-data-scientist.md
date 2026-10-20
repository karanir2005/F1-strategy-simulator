# T-015 + T-020 — data-scientist — 16-race tyre fit, ordered wear, fitted offsets

**Status:** done (pytest 308 passed)
**Next owner:** qa-engineer — verify; backend-engineer — see requests.

## What I did
- T-015: copied the six races (Barcelona, Madrid, Italy, Azerbaijan, Singapore, Netherlands) staging -> `data/processed/` (7 files each, extra columns season/era/source kept). Now 16 races. `tire_model` has an era filter: default `eras=(ERA_2026,)`; 2025 only via `build_tire_model(eras=(ERA_2025, ERA_2026), era_weights={ERA_2025: w})`. `clean_laps` also drops wet laps and stints running through a red flag (Monaco 222 laps, Italy/NL 36).
- T-020: no partial edits existed. `fit_joint_model(constrain=True)` = bounded LS with SOFT>=MEDIUM>=HARD>=0 wear and SOFT<=MEDIUM<=HARD pace; now the default. Offsets are fitted and used (placeholder only behind `fitted_offsets=False`). Cliff penalties stay PLACEHOLDER (not binding: max age ~50 laps).
- Finding: the free 2026 fit is inverted (MEDIUM slower than HARD in 99% of race-bootstrap resamples; SOFT +0.13, MEDIUM +0.19 s; wear .025/.035/.036). Not a pipeline bug: clean-air, lap fixed effects, pit-stop jump estimator all agree; the 2025 data gives the textbook order (SOFT -0.37, wear .061>.040>.036). So the constraint is imposed against the data. Shipped: offsets S -0.12, M 0.0, H 0; wear .037/.037/.035; fuel -0.0507 s/lap (sensible, ~1.7 kg x 0.03 s/kg).
- Rebuilt `data/tracks/track_params.json` (base laps moved by -0.7..+0.3 s; nothing else). New scripts: `calibrate_tire_compounds.py`, `check_optimizer_plans.py` (-> `data/scorecard/tire_compound_fit.json`, `optimizer_plans.json`). `validate_scorecard.run_scorecard` gained `model_kwargs`/`extra_train_laps`.

## Verification (before -> after)
| | before, 10 races | before, 16 races (old fit) | after, 16 races |
|---|---|---|---|
| per-stint MAE excl SC (s/lap) | 1.148 | 1.572 | 1.554 |
| relative MAE | 0.489 | 0.765 | 0.730 |
| finishing-order rho (grid) | 0.863 (0.854) | 0.869 (0.834) | 0.870 (0.834) |
| strategy-pair accuracy | 0.654 (n=136) | 0.690 (n=277) | 0.661 (n=277, CI .58-.74) |
- Honest read: all 16-race differences are inside noise. The 10->16 jump in MAE is the new races (Barcelona bias -2.9 s/lap from the pooled quali ratio), not the refit. Pair accuracy is down 3 points, not significant.
- validate_replay (leaky, excl. SC/VSC/red, abs %): before 0.02-1.03 -> after 0.13-1.10 (China .89->.69, Britain GAS .05->.13, Britain PIA 1.03->1.10, Austria .02->.37, Hungary .42->.84).
- Optimizer vs real: model-time gap between real top-10 plans and optimizer best (17 comparable tracks): median 15.8 s (old) -> 2.3 s (new); free fit 5.8 s.
- Pinned tests updated: test_t002_qa (original 10 only), test_scorecard (16), test_t005_qa (338 pairs, new fallbacks). New: `backend/tests/test_t020_ds.py`.

## Optimizer top plan vs winner (gap = model seconds winner's plan is slower than top plan)
| track | top | winner | gap | | track | top | winner | gap |
|---|---|---|---|---|---|---|---|---|
| australia | H34/S24 | M12/H46 | +11.8 | | azerbaijan | S27/H24 | M31/S5/S15 | +20.2 |
| china | M23/H33 | M10/H46 | +10.1 | | malaysia | M30/S25 | wet start | n/a |
| japan | S20/H33 | M22/H31 | +1.8 | | singapore | H30/S32 | S32/M30 | +0.8 |
| miami | H29/M28 | M26/H31 | +0.1 | | austin | H28/S28 | M33/S23 | +2.3 |
| canada | H33/S35 | S31/M37 | +1.8 | | mexico | H35/S36 | S34/M37 | +1.5 |
| monaco | S32/H46 | M37/H24/S7/S10 | red flag | | brazil | M36/S35 | M30/S20/M21 | +10.9 |
| barcelona | S21/S21/H24 | S11/H16/M14/H25 | +21.4 | | las_vegas | H26/M24 | M25/H25 | 0.0 |
| austria | H22/S24/S25 | M19/H24/H28 | +5.9 | | qatar | S20/S20/H17 | M7/M25/H25 | +8.5 |
| britain | H27/M25 | M25/H23/S4 | +25.8 | | abudhabi | H31/M27 | M23/H35 | +0.9 |
| belgium | H21/S23 | M18/S26 | +0.6 | | netherlands | H22/S25/S25 | M2/S19/H22/H3/H25 | red flag |
| hungary | H36/S34 | M17/H22/H17/S14 | +24.9 | | italy | H29/S24 | H3/M25/M25 | red flag |
| madrid | H34/S23 | M14/H43 | +8.8 | | | | | |

**Outliers, honestly:** the requested outcome is only partly met. 19 of 23 top plans are 1-stop, 4 are 2-stop, none has a stint under 10 laps (good). But 18 of 23 contain SOFT (mean 40% of laps) and 4 have two SOFT stints (Barcelona, Austria, Netherlands, Qatar). MEDIUM appears in only 6 top plans, because in this fit MEDIUM is dominated by HARD (same pace, slightly more wear) — that is what the 2026 data say, not tuning. Stop count matches the top-10 mode at 15/17 comparable tracks; compound set matches the winner at 5/17. Worst disagreements: Barcelona, Britain, Hungary, Azerbaijan (gap 20-26 s; those winners ran 3 stops or late short sprints). Plans also sit at the observed-maximum stint caps (e.g. S24 at Australia), so cap quality drives them as much as the tyre fit.

## Interface changes
`build_tire_model(...)` gains `eras`, `era_weights`, `constrain`, `fitted_offsets`, `drop_red_flag_stints`; model dict gains `_compound_offsets_unconstrained`, `_wear_unconstrained`, `_constrained`, `_eras`. Existing keys and `predict_delta`/`fuel_effect` unchanged. `load_all_laps(pattern, eras)`; new `load_2025_staging_laps()`.

## Requests / open items
- backend: `optimizer.ASSUMPTIONS` still says offsets are placeholders; reword (offsets fitted + ordering-constrained, cliffs placeholder). Consider robust stint caps (p90 rather than max) — I can add a `typical_max_stint_laps` field if wanted. `calibrate_lap_noise.py` / pit-loss constants were NOT rerun on 16 races.
- CLAUDE.md numbers to change: "currently 10 races" -> 16; replay error "0.02-1.03%" -> "0.13-1.10%"; tire_model offsets are now fitted (ordered), cliffs still placeholder.
- Not done: no deeper fix for the 2026 inversion (needs C-number allocations or FP long-run data); the 2025-pooled variant is available but not default (marginally better LORO: rho .875, pair acc .686).
