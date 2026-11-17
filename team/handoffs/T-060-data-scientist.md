# T-060 — data-scientist — calibrate the full-grid race sim

**Status:** done
**Next owner:** orchestrator — commit; decide on the recommendation below (do not wire the sim into /predict). Backend engineer: two stale text strings (below).

## What I did
- `calibrate_grid_sim.py` (new, repo root): fits every T-034 placeholder from real 2026 data as a function of a training set of races, so the same code gives the shipped constants (all 19 dry races) and the leave-one-race-out (LORO) fold constants. Writes `data/sim_params/grid_sim_calibration.json` (+ `data/scorecard/grid_sim_selection_table.csv`).
- `race_grid_sim.py` (additive only, no signature changes): `GridCalibration` + `load_grid_calibration()` (per-key fallback to the old placeholder constants, `source` / `placeholder_keys` like the interaction loader); `simulate_grid(..., calibration=None)` new optional last parameter (default = the JSON). Public signatures otherwise untouched, `plan_hook` (T-055) untouched.
- `grid_inputs.py` (additive): module constants `RACE_PACE_SCALE`, `DRIVER_SHRINK`, `QUALI_WEIGHT`, `PLAN_WEIGHTS`, `PIT_JITTER_LAPS` now come from the JSON (fallback = old values); new optional args `driver_shrink`, `plan_weights`, `pit_jitter`, `track` on `build_inputs` / `default_strategies` / `vary_plan`.
- `validate_grid_sim.py` rewritten: per-fold rebuild of everything the held-out race touched (leak fix, below), four configurations, standard errors, blends, probability scores, plus a 2025 season.
- Tests: new `backend/tests/test_t060_grid_calibration.py` (8: file loaded, per-key fallback, shipped file complete and plausible, default load used by the sim, overtake shift/scale used, held gap used, pace-noise constants used, grid_inputs constants equal the file). Two existing T-034 tests changed to pass `GridCalibration()` explicitly (they pinned the old placeholder spacing / pass rate); the faster-car test now asserts >0.95 with placeholders and >0.6 with the fitted file.

## Fitted constants (all 19 races; LORO range in brackets) and how
| Constant | Was | Now | Method |
|---|---|---|---|
| `RACE_PACE_SCALE` | 0.6 | **1.258** [1.23-1.29] | slope of real clean-air race pace (driver FE, race-lap FE, tyre/fuel normalised) on the sim's own pace input; R2 0.73 on 279 driver-races (outlier driver-races > 4 s from the race median dropped, Italy had a -37 s corrupt FE). Real race pace is spread MORE than the Q1 gaps, not less. |
| `DRIVER_SHRINK` | 0.5 | **0.25** [0.25-0.5] | grid search on regression SSE |
| `QUALI_WEIGHT` | 0.5 | **0.62** [0.59-0.70] | share of Q1 vs rating order in the same regression |
| `TEAM_PACE_SD_S` / `DRIVER_PACE_SD_S` | 0.20 / 0.15 | **0.43 / 0.33** | variance components of the regression residual, net of the pace-estimate sampling variance; the sim's realised pace spread (1.05 s sd) matches the real one |
| `SC_GAP_S` / `SC_GAP_CLOSURE` | 0.9 / 0.5 | **1.25 / 0.31** | floor and per-lap decay of the median car-to-car gap in real SC episodes. LOW CONFIDENCE: 20 late-SC laps, 5 decay ratios |
| `LAP1_SPACING_S` | 0.7 | **0.713** | slope of gap-to-leader at end of lap 1 per position, median over races (the placeholder was already right) |
| `PLAN_WEIGHTS` / `PIT_JITTER_LAPS` | 0.6/0.25/0.15, 2 | **0.52/0.26/0.22, 12 (cap)** | each real driver strategy assigned to the nearest optimizer top-3 plan; weights = shares; jitter = sd of matched green-flag stop laps (9.9 laps). Real fields are far more diverse than three plans +-2 laps; 12 is the cap I allowed. LOW CONFIDENCE, only a description of field diversity |
| overtake `held_gap_s`, `held_gap_sigma` (new) | min_follow_gap 0.31 | **median 0.871, lognormal sigma 0.519** | a car that fails to pass sits at median 0.87 s (n=3,609) next lap, not at the 5th-percentile floor 0.31 s. The old behaviour held every failed attacker at 0.31 s where the logistic gives 20-55% per lap, which is the main cause of the over-passing. Drawn from the unused part of the pass-attempt uniform, no new random numbers (CRN shapes unchanged). |
| overtake `pace_delta_scale`, `intercept_shift` (new) | 1, 0 | **1.0, -0.90** (folds -1.0..-0.85) | see below |

Pass-count calibration. Targets: real passes per race and the sd / mean-abs of grid-to-finish position change, none of them finishing-order accuracy, and the held-out race is excluded from the targets. Passes and position change lie on a ridge in (scale, shift): the coarse 2-D table (`data/scorecard/grid_sim_selection_table.csv`) shows pace scale 0 to 1.25 all reach the targets with a matching shift (e.g. 0.0/0.0, 0.5/-0.25, 1.0/-0.75 in an earlier run), so the pace scale is NOT identified by these statistics. I therefore keep the T-053 coefficient (scale 1) and fit only the intercept. A free 2-D search is scored as `calibrated_2d`; it gives the same answer within noise.

## The real pass count
T-053's "about 13 per race" is the 254 passes with a pace estimate for both cars. The same event definition without that requirement (needed only for the regression) gives 443 passes = **23.2 per race** over 19 races (R2-R19 for the target; R1 included 25), and that is what the sim must be compared to since it has no missing-pace cars. Sim 59.7-62.5 before, **22.0 after** (target 23.2). Per-race real counts range 3 (Monaco) to 49 (Italy); the sim sits at 12-35 and the per-race correlation sim vs real is -0.36 (held-out track offsets are 0 and nothing else predicts which races have passes), so the sim fixes the level not the race-to-race pattern.

## Validation leak
`track_params.json` (base lap, pit loss, stint caps) and `race_interaction.json` (overtake coefficients, track offsets, lap-1 table) both contained the held-out race. Fixed per fold in `calibrate_grid_sim.py` / `validate_grid_sim.py`: track entry rebuilt without the race (`rebuild_track`: base lap = median(race base lap / best Q1) of the other races x this weekend's best Q1, mean abs error 0.87 s = 1.0% vs the leaky value; pit loss and stints from the same-track 2025 race if dry else the other tracks' median), logistic + lap-1 table refitted on the other races with the held-out track offset dropped, all calibration constants refitted per fold, pass-calibration targets exclude the race. Effect of the leak, T-034 constants, same sims: Spearman 0.703 (leaky) -> 0.700 (fixed), top-3 0.579 -> 0.596, winner 0.579 -> 0.579, passes 62.5 -> 59.7. Negligible on finishing order. Residual leaks, small: the dirty-air loss, pass cost and min-follow-gap in `race_interaction.json` are not refitted per fold (one scalar each from ~15k laps); the shared strategy-mix statistics use per-race fold tyre models but pool other races whose models saw the held-out race; the (scale, shift) selection table is run with all-race constants and then re-tested with fold constants.

## Verification (LORO, 19 races R2-R20, n_sims 2000, mean +- se over races; `python validate_grid_sim.py`)
| | Spearman | top-3 hit | winner hit |
|---|---|---|---|
| grid order | 0.657 +- 0.039 | 0.596 +- 0.060 | 0.684 +- 0.110 |
| Plackett-Luce (walk-forward) | 0.702 +- 0.036 | 0.614 +- 0.064 | 0.684 +- 0.110 |
| sim, T-034 as committed (leaky) | 0.703 +- 0.035 | 0.579 +- 0.043 | 0.579 +- 0.116 |
| sim, T-034 leak fixed | 0.700 +- 0.036 | 0.596 +- 0.041 | 0.579 +- 0.116 |
| sim, fitted data constants (held gap in, no shift) | 0.710 +- 0.037 | 0.579 +- 0.043 | 0.684 +- 0.110 |
| **sim, calibrated (shipped)** | **0.707 +- 0.036** | **0.579 +- 0.035** | **0.684 +- 0.110** |
| 50/50 blend PL + sim (ranks by blended exp. points) | 0.708 +- 0.036 | 0.614 +- 0.053 | 0.684 +- 0.110 |

Paired, calibrated sim: minus grid **+0.050 (se 0.031, better in 11/19)**, minus PL **+0.005 (se 0.013, 12/19)**; blend minus PL +0.007 (se 0.008). All inside noise. The winner-hit gap to grid order (0.526 vs 0.684 in T-034) is gone (0.684, but one race is 5 points of the rate: se 0.11).

Realism, calibrated sim vs real (R2-R19): passes per race **22.0 vs 23.2** (was 59.7-62.5); sd of grid-to-finish position change **3.99 vs 3.99** (was 4.12-4.17); mean abs position change 3.33 vs 3.02 (was 3.4-3.5).

Probability quality (what /predict would show): P(win) log-loss on the actual winner sim **1.58 +- 0.22** vs PL **1.05 +- 0.22** (blend50 1.21); podium Brier sim 0.073 vs PL 0.065 (blend50 0.066). The sim's probabilities are clearly more diffuse than PL's.

Per season. 2026 is above. 2025 (13 races: all dry 2025 races on tracks with 2026 data; Miami skipped on a NaN, wet races and tracks whose track_params came from the 2025 race itself skipped), every constant is the all-2026 fit so the sim never saw 2025: sim 0.655 +- 0.059, grid 0.681 +- 0.061, PL 0.657 +- 0.065; top-3 0.744 / 0.821 / 0.821; winner 0.615 / 0.923 / 0.846. Sim minus grid -0.026 (se 0.019, better in 5/13). Passes 22/race (no real count for 2025). Position-change sd 4.9 vs real 3.6: the 2026 pace spread overshoots the tighter 2025 field. Caveat in the other direction: PL's config was tuned on the 2025 walk-forward, so 2025 favours it. So the sim's 2026 edge over grid order does not replicate in 2025.

Full suite: `pytest` 426 passed (143 s). Not run: `validate_replay.py` (tire_model.py and race_simulator.py untouched).

## Recommendation: do NOT feed the sim into /predict, not even as a blend
- Finishing order: no demonstrated gain over the Plackett-Luce predictor in 2026 (+0.005, se 0.013) and no gain over the grid order in 2025 (-0.026). The 50/50 rank blend moves Spearman +0.007 (se 0.008) and top-3 not at all.
- Probabilities: sim P(win) log-loss is 0.5 worse than PL (and a blend is still worse than PL alone). /predict shows probabilities, so this is the metric that matters, and it is a clear loss.
- Where the sim does add value is what PL cannot do: it responds to strategy, SC timing and pit calls (T-055). Keep it as the strategy-what-if engine, using its position distributions for that purpose, not as the winner predictor. Levers if someone wants to push P(win) quality: pace noise is the likely cause of the diffuseness (fitted 0.43/0.33 s, matches the real residual so lowering it should be argued from data, not from the scorecard), and a grid-slot (pole) term.

## Interface / contract changes
- `simulate_grid(..., calibration: Optional[GridCalibration] = None)` (new last optional parameter). `GridCalibration` fields: `team_pace_sd_s, driver_pace_sd_s, sc_gap_s, sc_gap_closure, lap1_spacing_s, ov_pace_scale, ov_intercept_shift, held_gap_s (None = old behaviour), held_gap_sigma, race_pace_scale, driver_shrink, quali_weight, plan_weights, pit_jitter_laps, source, placeholder_keys`. `load_grid_calibration(path=None)`.
- `grid_inputs.build_inputs(..., driver_shrink=None, plan_weights=None, pit_jitter=None, track=None)`; `default_strategies(..., plan_weights=None, pit_jitter=None)`; `vary_plan(..., jitter=None)`. All default to the old behaviour/new constants.
- Behaviour change for every caller using defaults (/simulate/grid, MCP `simulate_grid`, T-055 pit policy): the default constants are the fitted ones. Pace spreads are about 2x wider (scale 1.26 vs 0.6), passing is much less frequent (22 vs 60 per race), strategy diversity is larger, SC bunching looser. Any T-055 numbers produced before this change will move; its validation should be re-run.
- New file `data/sim_params/grid_sim_calibration.json` (layout in `load_grid_calibration`'s docstring), re-create with `python calibrate_grid_sim.py` (~6 min incl. the selection sims).

## Requests for other roles
- backend-engineer: (1) `api/routes/grid_sim.py:100` and `mcp_server.py:423` still say the quali-to-race-pace scaling and shrinkage are "judgement placeholders, not fitted": now fitted, please update the wording and consider returning `calibration_source` / `calibration_placeholder_keys` next to `interaction_source`. (2) Re-run `validate_pit_policy`/T-055 numbers with the new defaults (or pass `calibration=GridCalibration()` to reproduce the old ones). (3) Known engine gap seen while calibrating: a "free" car (outside the 1.5 s window at lap start) that ends the lap ahead of the car in front swaps places in the sort with no pass event, no pass cost and no count, so some real passing is unmodelled; not fixed.
- orchestrator: re-run `python calibrate_grid_sim.py` and `python validate_grid_sim.py` after any change to the pulled race set, the tyre model or the interaction parameters.

## Open questions / risks
- 19 races (13 in 2025): every difference above is within 2 se except the pass count and spread calibration, which are matched by construction.
- SC bunching and strategy-mix constants rest on thin / indirect evidence (see table). Pace-noise and pace-scale are the most consequential and are well supported (279 driver-races, stable across folds).
- Pace scale 1.26 is a 2026-regs fact (a more spread field); the 2025 replication shows the spread does not carry over to other eras. Refit each season.
- The pace-scale / intercept pair is not separately identified by pass counts (ridge); the choice of scale = 1 is a convention, not a measurement.
