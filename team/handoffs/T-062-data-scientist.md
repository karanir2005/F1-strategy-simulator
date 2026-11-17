# T-062 — data-scientist — full-grid sim under strict walk-forward

**Status:** done, negative result. No default changed; `data/sim_params/grid_sim_calibration.json` untouched.
**Next owner:** orchestrator (commit; nothing for other roles to consume).

## What I did
- New `grid_sim_walkforward.py` (repo root) and flag `python validate_grid_sim.py --walk-forward [--n-sims 2000]` (writes `data/scorecard/grid_sim_walkforward.json`). For round R EVERYTHING comes from rounds < R: tyre model (laps of rounds < R only), track entry (`rebuild_track` restricted to earlier races), overtake logistic / lap-1 table / track offsets, pace regression and noise, SC bunching, lap-1 spacing, held gap, overtake intercept shift (targets of earlier rounds). PL numbers are the existing walk-forward file. T-060's LORO used later races too.
- Evaluated rounds 5-19 (15 races). R1-4 are excluded: no usable tyre model / pace regression yet. Round 20 has no lap data. So n = 15, not 19, and numbers differ slightly from T-060 for that reason as well as for the strict fitting.
- Not walk-forward (small leaks, stated): the strategy mix uses the shipped T-060 values in every fold (it needs a tyre model per training race); the (scale, shift) selection table was built with all-race constants (only the shift choice uses it, restricted to rounds < R); the pace-table base lap uses the track entry's base lap as a unit conversion only.
- No code changes to `race_grid_sim.py`, `grid_inputs.py` or `predictor.py` (imported read-only). Variants replace the cars' `pace_delta_s` after `build_inputs`, so the experiments are opt-in and the old behaviour is the default.

## Re-baseline, strict walk-forward, n_sims 2000, mean +- se over 15 races (R5-R19)
| | Spearman | top-3 hit | winner hit | P(win) log-loss | podium Brier | passes/race |
|---|---|---|---|---|---|---|
| grid order | 0.655 +- 0.043 | 0.600 +- 0.067 | 0.600 +- 0.131 | - | - | - |
| Plackett-Luce | 0.700 +- 0.041 | 0.622 +- 0.072 | 0.600 +- 0.131 | **1.214 +- 0.268** | 0.063 +- 0.009 | - |
| sim, shipped (T-060 form, fitted WF) | 0.699 +- 0.044 | 0.533 +- 0.044 | 0.533 +- 0.133 | 1.745 +- 0.295 | 0.075 +- 0.004 | 23.3 (real 22.3), per-race corr -0.19 |
| sim, `theta_grid` pace input | 0.695 +- 0.041 | 0.622 +- 0.044 | 0.667 +- 0.126 | 1.391 +- 0.428 | 0.068 +- 0.007 | 21.5, corr -0.22 |

Paired differences (sim minus X, mean +- se of the per-race difference):
| | Spearman | top-3 | winner | win log-loss | podium Brier |
|---|---|---|---|---|---|
| shipped sim - PL | -0.000 +- 0.015 | -0.089 +- 0.083 | -0.067 +- 0.182 | +0.531 +- 0.271 | +0.012 +- 0.008 |
| shipped sim - grid | +0.044 +- 0.024 | -0.067 +- 0.081 | -0.067 +- 0.182 | | |
| theta_grid - PL | -0.005 +- 0.010 | 0.000 +- 0.056 | +0.067 +- 0.118 | +0.177 +- 0.268 | +0.005 +- 0.005 |
| theta_grid - grid | +0.039 +- 0.017 | +0.022 +- 0.051 | +0.067 +- 0.118 | | |
| theta_grid - shipped sim | -0.003 +- 0.009 | +0.044 +- 0.055 | +0.067 +- 0.153 | -0.324 +- 0.218 (z -1.5) | -0.007 +- 0.005 (z -1.5) |

Strict walk-forward does not change the picture from T-060: Spearman equals PL, the sim is better than grid order by about 0.04 (about 2 se for theta_grid, 1.8 for the shipped one), and the sim's P(win) is the weak spot.

## Experiments (1000 sims per config, same 15 races; full table in the run log, key rows here)
(a) PL theta as pace input. `theta`: real race pace regressed (rounds < R) on the shrunk Q1 gap (s) plus the PL theta_race gap to the best driver (rating units), coefficient fitted per fold, replacing the quantile-matching blend. Spearman 0.704 vs 0.701, win LL 1.61 vs 1.71, top-3 0.578 vs 0.556. All inside noise. It does lower the fitted noise (team 0.30 / driver 0.28 s vs 0.43 / 0.33) and raises R2 (0.81 vs 0.73 in-sample), but that does not translate into finishing-order accuracy.

(b) Noise / scale tuning. Pace-noise multiplier 0.4, 0.6, 0.8, 1.0, 1.25, 1.5 and pace multiplier 0.8, 1.0, 1.2 on base and theta_grid. Win log-loss gets WORSE as noise falls (base: 1.71 at 1.0, 1.83 at 0.6, 1.90 at 0.4; theta_grid flat 1.37-1.43 for multipliers 0.8-1.5). Position log-likelihood is flat (2.61-2.71) across everything. So the diffuseness is NOT from over-large pace noise; T-060's lever (lower noise) is refuted. Pass count stays within +-25% of real (20-26) for nearly all configs. A fully walk-forward selection over all configs (loss = per-driver position log-likelihood of earlier races, passes within +-25%) gives Spearman 0.692, win LL 1.57, i.e. it ends up near theta_grid and is no better than the fixed choice.

(c) Track overtake prior from the same track's 2025 passes. Checked directly on the 13 tracks with both seasons (T-053 event definition): Pearson of log(1+passes) 0.64, driven by Monaco (2 vs 3 passes); without Monaco -0.41; Spearman -0.21. No usable predictive signal beyond "Monaco is hard". Not implemented. Per-race pass correlation remains negative (-0.19 to -0.26) for every config tried.

(d) Grid-slot term. `theta_grid` adds log(grid slot) as a regressor in the pace regression (coefficient fitted per fold on rounds < R). It is the one thing that moved a number: the sim's mean P(win) for the pole sitter rises from 0.24 to 0.48 (PL 0.55, actual pole-win rate 0.60), win log-loss falls 1.71 -> 1.38-1.39, winner hit 9/15 -> 10/15, top-3 0.556 -> 0.622. A linear (grid-1, capped at 9) term is worse (win LL 1.52), `quali_grid` (no theta) 1.43. This confirms the sim underweights track position: its dirty-air / overtake model lets the faster car through too often relative to the pole's real advantage, and the pace-regression term soaks that up as an empirical patch (it is not a physical mechanism).

## Blend with PL (probabilities, weight w on the sim, chosen walk-forward by P(win) log-loss of earlier races, 0.5 for the first three)
- Shipped sim + PL: the chosen weight goes to 0 from round 8 on; fixed-w win LL rises monotonically from 1.214 (w 0) to 1.707 (w 1). The sim adds nothing to PL. WF blend 1.235 +- 0.260, +0.021 +- 0.039 vs PL.
- theta_grid sim + PL: fixed-w win LL 1.214 (0), 1.187 (0.3-0.4), 1.192 (0.5), 1.384 (1). In-sample best w 0.3-0.4 improves 0.03 only; the walk-forward weight wanders (0, 0.3, 0.5, 1.0), WF-chosen blend is 1.387, +0.17 +- 0.22 vs PL (worse, noise). No blend weight can be recommended.

## Decision
Nothing beats the 2 se criterion on a paired difference against PL or against the shipped sim (best: theta_grid vs shipped, win log-loss -0.32 +- 0.22, z -1.5; podium Brier z -1.5). So I did NOT change the shipped calibration, and no code path was added to the default behaviour. `grid_sim_walkforward.py` keeps the `theta_grid` variant reachable (`run_walk_forward(variants=("base","theta_grid"))`) as an opt-in experiment, not wired to `simulate_grid` or `grid_inputs`.

If the orchestrator wants to act on it anyway: the evidence is suggestive for a grid-slot / track-position term (it fixes the pole-sitter probability, the one visible miscalibration); next step would be a physical version in the engine (a lap-1 and early-race track-position advantage, or an overtake probability that is lower for the first N laps), not a pace-regression patch, and judged on 2025 as well. Recommendation for /predict is unchanged from T-060: keep PL for probabilities; the sim is the strategy what-if engine.

## Caveats
- 15 races; every difference above except sim-vs-grid Spearman (about 2 se) is within noise.
- Selection among about 25 configs on 15 races overstates the best one; theta_grid was one pre-planned variant of five, and the walk-forward selection that accounts for this did not do better than it.
- Early folds (R5-R7) fit the pace regression on 3-5 rounds (about 60-100 driver-races); the tyre model has 4-6 races.
- Runner time: about 45 s for the strict walk-forward at 2000 sims (study build is cached tyre fits). `pytest`: 438 passed, 3 xfailed. `validate_replay.py` not run (tire_model / race_simulator untouched).
