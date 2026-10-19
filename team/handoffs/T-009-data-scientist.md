# T-009 — data-scientist — Race set expansion via OpenF1 + Jolpica

**Status:** done (staging only; awaiting QA before anything moves to `data/processed/`)
**Next owner:** qa-engineer — verify the loader and staging data (see "For QA"). Then orchestrator decides the promotion policy (open questions).

## What I did
- Rebuilt the expansion on **OpenF1 + Jolpica** (FastF1 not used, `cache/` untouched). Reused the partial `openf1_loader.py` fetch layer and the raw cache, then finished the convert/build layers.
- **41 races staged**: all 24 of 2025 and 17 of 2026 (Australia, China, Japan, Miami, Canada, Monaco, **Barcelona** R7, Austria, Britain, Belgium, Hungary, Netherlands, Italy, **Madrid** R14, Azerbaijan, **Malaysia** R16 = "Bahrain GP" held at Sepang, Singapore). Bahrain/Saudi 2026 are cancelled and skipped. Labels: `CIRCUIT_LABEL` in the loader.
- Output in `data/staging/`: `{label}_{season}_laps.csv`, plus `_results`, `_weather`, `_track_status`, `_rcm`, `_pit_stops`, `_race_info.json` and a new `_quali.csv` (Jolpica Q1-Q3). 328 files, 14 MB. Raw API cache is `data/openf1/raw/` (182 MB, includes `intervals`/`position`, Jolpica results/qualifying/pitstops/schedules).
- **Schema** = `extract_laps_df` columns, same order, plus `season`, `era` (`2025_regs` / `2026_regs`) and `source` on every laps and sidecar row. `race_info.json` also carries `season`, `era`, `round`, `openf1_session_key` and a `data_quality` block.
- The conversion hands a FastF1-named frame to `data_loader`'s own helpers (`_assign_tyre_stints`, `_decode_track_status`, `_add_gaps`, `_add_overtake_enabled`, `_add_weather`), so "real tyre change", gaps and Overtake logic have one definition for both sources.
- Track status is rebuilt from race control (SC deployed to track clear, VSC deployed/ending, red to restart, sector yellows) and intersected with each lap's own window, written as FastF1-style code strings.
- New `pull_openf1.py` (CLI: `--only 2026:Japan`, `--out`, `--today`). Deleted `_fetch_all_tmp.py`. `pull_races.py` comment fixed (Madrid vs Barcelona).
- Tests: `backend/tests/test_openf1_loader.py` (15 tests, offline plus one real-cache build). **`pytest`: 256 passed.**

## Files changed
- `backend/app/core/openf1_loader.py` — rewritten/completed (fetch, convert, quality checks, build, write).
- `pull_openf1.py` — new CLI. `pull_races.py` — comment only. `_fetch_all_tmp.py` — deleted.
- `backend/tests/test_openf1_loader.py` — new.
- `data/staging/*` (new), `data/openf1/raw/*` (cache). Nothing in `data/processed/` touched.

## Verification — OpenF1 vs FastF1 on the 10 overlapping 2026 races
Row-level join on (driver, lap_number); the full harness is in my scratchpad (`cmp.py`).
- **Lap rows:** identical set in China, Japan, Miami, Austria, Britain, Belgium (0 only-in-either). Canada/Monaco/Hungary lose 2/3/1 rows (OpenF1 lacks a lap). **Australia is the one broken race** (below).
- **Lap times:** 97.4-98.5% of rows agree to < 0.05 s; p95 absolute difference 0.00 s. The rest are NaN in one source: OpenF1 times some SC/red laps FastF1 leaves NaN (0.2-6% of rows), never the reverse (except Australia).
- **Lap 1** is systematically **+0.37 to +0.45 s slower in OpenF1** (standing-start reference). Lap 1 is already excluded from fitting. Not corrected.
- **Compound:** 100% agreement in 6 of 10 races; 97-99.9% elsewhere. Disagreements: mid-race stints OpenF1 has with `compound=None` (Hungary NOR 17 laps, left as `tyre_data_missing`), and Belgium ANT (OpenF1 HARD, FastF1 SOFT for the same stint).
- **Tyre age (laps on set):** 100% in China/Japan, 95-99% elsewhere. All differences are in start age of used sets (FastF1 vs OpenF1 disagree on whether the set was new: e.g. Hungary HUL SOFT 4 vs 1, Britain HAD 4 vs 1, Canada PIA MEDIUM 5 vs 1, Miami ALB lap-1 set 1 vs 4). Neither source is verifiably right; OpenF1 reports **lower** start ages more often (Austria/Britain/Belgium/Hungary/Canada) and higher once (Miami). Effect: used-set stints look fresher in OpenF1 data.
- **Stints per driver (same compounds and start ages):** China 18/18, Japan 22/22, Miami 21/22, Austria 21/22, Canada 20/21, Britain 19/22, Belgium 19/21, Hungary 18/22; Monaco 12/21; Australia 12/20.
- **Tyre changes / same-tyre pit passes:** FastF1 vs OpenF1 totals: China 19/19, Japan 26/26, Miami 21/21, Canada 33/33, Austria 41/41, Britain 50/50, Hungary 45/45, Belgium 28/27; Monaco 51/67 and passes 35/19 (see below).
- **Pit laps:** in-lap match 99.7-100%, out-lap 99.9-100%. `pit_lane_s`: median difference within +/-0.06 s, p95 abs 0.14-0.21 s (n=19-86 per race).
- **Track status flags (per lap):** has_sc 100% except Monaco (97.2%); VSC 99.7-100%; red 99.4-100%; yellow 99.9-100%; `is_green_flag` 99.8-100% except Monaco (97.6%).
- **Position** equal on 99.3-100% of laps; **gap_ahead_s** median absolute difference 0.05 s (lap-end timestamps differ slightly).
- **Independent check vs Jolpica pit-stop table (all 41 races):** 40 of 41 races match 100% of Jolpica (driver, lap) stops; Hungary 2026 44/45; Miami 2025 12/18 (broken data). This is the validation available for 2025, where there is no FastF1 reference.

### Systematic differences / OpenF1 quirks (all handled in the loader unless noted)
1. **Stint boundaries on the in-lap** in some races (Canada, Monaco, Hungary...): OpenF1 `stints.lap_start` equals the pit in-lap, which makes ages +1 and the boundary a lap early. Fixed (`normalise_stints`); Canada age agreement went from 30% to 97%.
2. **Trailing 1-lap `compound=None` stint** on retirements creates phantom tyre changes. Merged into the previous stint.
3. **Extra untimed row** after a car's last completed lap (retirement lap or post-chequered in-lap). Dropped using Jolpica laps.
4. **Duplicate positions** in the position stream (momentary swaps). Re-ranked per lap by lap-end time so `_add_gaps` works.
5. **Lap 1** often has no `lap_duration`; filled from the lap-2 start. `is_accurate` is NaN (OpenF1 has no equivalent; nothing reads it). `deleted` is parsed from race-control "LAP DELETED" messages.
6. **Time base:** `lap_start_s`, `lap_end_s`, weather and track-status `time_s` are seconds since lights out (earliest lap-1 `date_start`), not FastF1 session seconds. Only differences are meaningful.
7. **`status`** in the results sidecar is Jolpica's vocabulary (`Finished`, `Lapped`, `Retired`, `Accident`...) and `gap_to_winner_s` is only set for `Finished` cars on the lead lap.
8. **Not fixed, flagged:** Monaco 2026 red flag. OpenF1 splits post-red stints with implausible consecutive new sets (+16 tyre changes vs FastF1). Treat Monaco stints as unreliable around the red flag.
9. **Australia 2026 is unusable from OpenF1:** its lap counter collapses (OpenF1 "lap 1" spans FastF1 laps 1-3; later lap numbers are off by one; the first stint compound is None). `usable=false`. Keep the FastF1 version in `data/processed/`.

### Data quality over the full set (`race_info.json` -> `data_quality`)
- **39 of 41 races `usable=true`** (self-coverage >= 0.97, inner lap gaps <= 1%, finishers' last lap timed).
- **Unusable:** `Australia_2026` (above) and **`Miami_2025`** (OpenF1 is missing ~45% of laps: laps 2-24 absent for every driver; only 559 rows vs ~1,100 expected).
- Netherlands 2026: 3 missing laps (HUL, NOR, TSU), still usable.
- Every other race: self-coverage 1.000, 0 inner gaps.

## Interface / contract changes
- New staging files and columns: `season`, `era`, `source` on laps and sidecars; `_quali.csv`; pit_stops sidecar gains `stop_duration_s` (stationary time from OpenF1); `race_info.json` gains `season`, `era`, `round`, `openf1_session_key`, `data_quality`.
- Differences from the FastF1 CSVs a consumer must know: `compound_c` is only filled for the 10 original 2026 events (`COMPOUND_ALLOCATION`); all other races are NaN, never guessed. `event_format` is "unknown". `is_accurate` NaN.
- `tire_model` is unchanged and only globs `data/processed/*_laps.csv`, so staging cannot leak into fitting.

## Requests for other roles
- qa-engineer: see below.
- orchestrator: decide promotion (open questions), and add `compound_c` allocations for the 31 uncovered events if wanted (a researcher task; Pirelli nominations are public).

### For QA
- Re-run `python pull_openf1.py` (offline from cache, byte-identical output expected) and `pytest backend/tests/test_openf1_loader.py`.
- Spot-check `Monaco_2026` and `Canada_2026` stint tables against `data/processed/` (comparison harness path: scratchpad `cmp.py`).
- Confirm no staging file is read by `tire_model` or `predictor`.
- Check `Miami_2025` / `Australia_2026` are marked `usable=false` and that nothing downstream should consume them.

## Open questions / risks
- **Used-set start age disagreement** (FastF1 vs OpenF1) affects ~3-5% of rows in most races. For fitting wear, prefer OpenF1 vs FastF1 only after a leave-one-race-out check; for 2026 races already in `processed`, I recommend keeping FastF1 as primary and using OpenF1 only for the other 31.
- **Era mixing:** 2025 rows are tagged `2025_regs`. Per the roadmap, a model that mixes eras must beat a 2026-only model in walk-forward CV.
- **Qualifying** is Jolpica Q1-Q3 only (grid and pace). I did not pull OpenF1 qualifying lap-by-lap sessions; the Qualifying session keys are already in `sessions_{year}_Qualifying.json` if wanted.
- **Not verified:** `overtake_enabled` per-lap truth; 2025 tyre ages against any second source (only pit laps were cross-checked); OpenF1 `intervals` are cached but unused (gaps are recomputed from lap-end times to match FastF1's definition).
- OpenF1 and Jolpica are community APIs; raw responses are cached, so reruns never hit the network.
