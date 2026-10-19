# T-003 — data-scientist — Validation scorecard v2

**Status:** done
**Next owner:** qa-engineer — verify (leakage, metric definitions, the Jolpica path fix). Then orchestrator: T-005 can use `data/scorecard/latest.json` as the Phase 0 baseline.

## What I did
- Resumed the partial `validate_scorecard.py` (did not restart). The only defect was `load_jolpica_qualifying`: it globbed `2026_*_qualifying__o0.json`, a pattern that matches neither what the client writes for 2026 (`2026_<round>_qualifying__o0.json`, per round) nor the old hardcoded season file, so it silently found nothing (the T-012 QA finding). It now reads the client's per-round cache files, named via `jolpica_client._cache_path` / `PAGE_LIMIT`, with the season-wide file as a last fallback. It is read-only: no HTTP, no mkdir, no writes (`get_pages(offline=True)` was not used because it mkdirs its module-level `RAW_DIR` and the tests can't redirect it).
- Fixed `backend/tests/test_scorecard.py` to the per-round file layout (fixture now named via the client and carries `MRData.total`). 18 tests pass; full suite 222 passed.
- Ran the scorecard end to end on the 10 cached 2026 races and wrote `data/scorecard/latest.json` (deterministic, about 6 s, no network, no FastF1).

Design (details in the script docstring): every race is scored with a tyre model fitted on the other 9 races (LORO) and a non-leaky `base_laptime` = driver's best Q1/Q2/Q3 time (Jolpica) x a pooled race/quali ratio fitted only on the 9 training races (`quali_ratio`, about 1.038). `fastest_lap` (the old `validate_replay` method) and `oracle` (base fitted on the scored race's own clean laps) are leaky reference rows only. `test_loro_and_quali_ratio_never_see_the_scored_race` corrupts one race's laps and checks that race's model, ratio and finishing-order rho are bit-identical.

## Baseline scorecard (10 races, LORO, `quali_ratio`)

Per-stint s/lap error (sim - actual; lap 1 and pit in/out laps excluded; n = stints with >= 3 laps):

| base method | excl SC/VSC/red (headline) MAE / bias / relative MAE | incl SC/VSC/red MAE / bias | clean laps MAE |
|---|---|---|---|
| quali_ratio (non-leaky) | **1.148 / -0.470 / 0.489** (n=462) | 4.375 / -4.042 (n=490) | 1.115 |
| fastest_lap (LEAKY) | 0.851 / +0.137 / 0.489 | 3.943 / -3.454 | 0.814 |
| oracle (LEAKY floor) | 0.525 / -0.228 / 0.489 | 3.942 / -3.802 | 0.474 |

"Relative MAE" removes each driver's mean stint error, so the base level cancels. It is the same 0.489 for all three methods and is the part that matters for comparing two strategies for one car. Median |err| (quali_ratio, headline) 0.856.

By compound (quali_ratio, excl neutralised): HARD MAE 0.997 (n=186), MEDIUM 0.923 (n=191), SOFT 1.986 (n=85, relative MAE 0.680; oracle SOFT 1.262). By race: MAE 0.54 (Japan) to 2.61 (Hungary, bias -2.61 s/lap; Hungary's real race/quali ratio is 1.070 vs the pooled 1.038).

Finishing-order Spearman vs actual (171 classified finishers, mean over 10 races):

| predictor | mean rho |
|---|---|
| grid order (baseline) | 0.855 |
| qualifying order | 0.882 |
| one-car sim, quali_ratio | **0.865** (+0.010 vs grid, race-bootstrap 95% CI -0.030 .. +0.047; beats grid in 6/10 races) |
| one-car sim, fastest_lap (LEAKY) | 0.767 |
| one-car sim, oracle (LEAKY) | 0.903 |

Strategy-ranking accuracy (classified finishers, quali best within 0.5% of each other, different compound sequence / stop count; "who finished ahead"):

| set | n pairs (races) | sim | strategy-only sim | grid order |
|---|---|---|---|---|
| primary | 136 (10) | **0.662** (race-bootstrap95 0.54 .. 0.745) | 0.618 | 0.632 |
| broad (also same sequence, stops >= 5 laps apart) | 190 (10) | 0.674 | 0.611 | 0.637 |

Sim minus grid (primary): +0.029, bootstrap95 -0.15 .. +0.16, so not distinguishable from the grid yet. Sim minus strategy-only: +0.044 (-0.007 .. +0.094).

Race totals, Finished drivers (mean |%| error): all laps 7.15% (the engine has no safety cars; bias -7.06%); excluding SC/VSC/red laps 0.92% (quali_ratio), 0.63% (fastest_lap, leaky), 0.32% (oracle).

Read-out for Phase 0/1: the shape model (relative MAE about 0.5 s/lap, SOFT worst) is decent; the dominant error is the per-track pace level (the cost of not leaking is +0.30 s/lap MAE vs `fastest_lap`, +0.62 vs the oracle floor). The one-car sim does not beat qualifying order for the finishing order (0.865 vs 0.882), and is only +0.01 over grid, so the Phase 1 exit criterion ("clearly beats grid") is not met yet, as expected. Neutralised laps (SC/VSC/red) add about 3.2 s/lap of MAE, the case for Phase 2.

## Files changed
- `validate_scorecard.py` (new, repo root) — the scorecard; Jolpica qualifying loader fixed to the client's per-round cache files.
- `backend/tests/test_scorecard.py` (new) — 18 tests (pure helpers, qualifying loader, LORO no-leak check, determinism, JSON schema).
- `data/scorecard/latest.json` (new) — the baseline above, with per-pair rows, per-race/compound tables and the LORO model coefficients used for each race.

## Verification
- `pytest backend/tests/test_scorecard.py`: 18 passed. Full `pytest` from repo root: 222 passed (54 s).
- `python validate_scorecard.py`: runs end to end, writes `data/scorecard/latest.json`; the numbers above are from that run. Two runs give identical JSON (apart from `generated_at`; tested).
- Not verified: independence of the strategy pairs (they share drivers, so the Wilson CI is optimistic; use the race bootstrap); I did not hand-check individual pair labels beyond the structural test. Quali time for drivers knocked out in Q1 is a Q1 lap on an earlier, slower track than Q3, which adds some noise to the pace level and to the similar-pace pairing. Canada (7 drivers, INTERMEDIATE stints) and Hungary PER are skipped as unsimulatable (in the JSON under `meta.skipped_drivers`). Australia VER/SAI/STR and Miami HAD have no qualifying time and use the teammate's (flagged under `base_laptime.*.quali_fallback_drivers`).

## Interface / contract changes
- none to product code. `validate_scorecard.run_scorecard(gps=None, processed_dir=..., raw_dir=...)` returns the card dict; `python validate_scorecard.py [--out PATH] [--no-write]`.

## Requests for other roles
- orchestrator: re-run `python validate_scorecard.py` after any change to `tire_model.py` or `race_simulator.py` and compare `headline` (per-stint MAE/relative MAE, finishing-order rho, strategy-ranking accuracy). `validate_replay.py` still uses the leaky fastest-lap `base_laptime`; treat the scorecard as the number to track and `validate_replay` as a smoke test (I did not edit it).
- data-scientist (later, tire_model lane): the SOFT curve is the weakest part (relative MAE 0.68 vs 0.24/0.31 for HARD/MEDIUM). No `tire_model.py` change requested for T-003.

## Open questions / risks
- The pooled race/quali ratio ignores track; per-track level is the biggest error source. Options for Phase 1: a track-pace prior from earlier seasons' races at that circuit, or practice long runs (OpenF1).
- The sample is small (10 races, 136 pairs), so strategy-ranking differences of a few points are not significant yet. T-009 (more races) would tighten the intervals.
- The 10 races include Hungary/Austria with unusually high race/quali ratios; a leave-one-race-out check of the ratio's stability is in `base_laptime.*.quali_ratio` vs `in_race_oracle_ratio_median` in the JSON.
