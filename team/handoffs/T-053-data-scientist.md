# T-053 — data-scientist — race-interaction parameters for the full-grid sim

**Status:** done
**Next owner:** backend-engineer — consume `data/sim_params/race_interaction.json` in `race_grid_sim.py` (T-034). Contract below.

## What I did
- Fitted dirty air, overtake probability, pass time cost, lap-1 position changes and minimum following gap from 19 dry 2026 races (18 from `data/processed`, Malaysia from `data/staging`; wet races dropped with `drop_wet_races`). The laps CSVs already carry position and on-road gaps, so no new pulls were needed.
- All gaps are end-of-lap timing-line gaps (`road_gap_ahead_s`, same-lap cars only), one sample per lap. The sim must use the same definition (gap at the end of lap n).
- Dirty air: two-way fixed effects (stint, race-lap) on clean laps. Controls tyre age, fuel and driver pace. Race-bootstrap CIs.
- Overtake: penalised logistic on (attacker, lap n) observations with a car directly ahead within 1.5 s. Success means the attacker crosses the line ahead of that car on lap n+1 with no pit in/out, SC/VSC or lapped car involved.

## Files changed
- `fit_race_interaction.py` (new) — fitting script, run from repo root, `--no-write` and `--quick` flags, ~15 s.
- `data/sim_params/race_interaction.json` (new) — output.
- `backend/tests/test_race_interaction_params.py` (new) — schema + sign tests (5 pass).

## Verification
Sample sizes: 14,891 clean laps (3,222 within 1.0 s of the car ahead); 2,335 overtake observations with 254 passes (base rate 0.109 per lap); lap 1: 408 drivers; 5,388 sustained-follower laps.

| Parameter | Value | Evidence |
|---|---|---|
| dirty_air.window_s / time_loss_s | 1.0 / **0.31 s/lap** (se 0.05) | By gap bin vs clean: <0.5 s +0.40, 0.5-1 s +0.19, 1-1.5 s +0.04, 1.5-2 s -0.01. Sustained (2 consecutive laps) gives 0.31 as well. |
| dirty_air.extra_deg_frac | **0.0 (fallback)** | Raw estimate -0.30 (95% CI -0.79..+0.01): cars in dirty air wear tyres *less* (they run slower), not more. No extra wear detectable, so 0. |
| overtake logistic | intercept -0.04, pace_delta **+1.31**/s, tyre_age_delta **+0.079**/lap, gap **-4.35**/s | 95% race-bootstrap CIs: pace 0.96..1.96, age 0.04..0.12, gap -5.2..-3.7. |
| pass_time_cost_s | attacker **+0.12** (se 0.07), defender **+1.05** (se 0.09) | 443 passes in the cost regression. Extra lap time on the lap the pass is recorded, vs clean air, beyond the dirty-air term. |
| lap1 (mean, sd) by grid bin | 1-3 (+0.47, 1.77), 4-6 (+0.37, 3.40), 7-10 (+0.64, 3.43), 11-15 (-0.82, 2.63), 16-22 (-1.09, 2.55) | Positive = lost places. n = 55/54/74/94/131. |
| min_follow_gap_s | **0.31** | 5th percentile of sustained follower gaps (1st pct 0.13, median 0.91). Timing-line resolution. |

**Held-out check (leave-one-race-out over 19 races, held-out track offset = 0):** log-loss model **0.2395** vs constant-rate baseline **0.3378** (diff -0.098 +/- 0.018 se across races; model better in 17 of 19, worse in Madrid and Monaco which have 4 and 1 passes). Ablations vs const: gap only -0.073, pace only -0.037, tyre age only -0.011, pace+gap -0.093. So gap and pace carry the signal and tyre age adds a little.

**2025 prior:** adding 2025 staging at weight 0.3 changed LORO log-loss by -0.001 +/- 0.002, i.e. no gain, so 2026-only is used (`used_2025_prior: false`).

**Track offsets:** can't be tested leave-one-race-out (each track appears once in 2026). Tested by transfer instead: fit on 2026, score same-track 2025 races (2,201 obs): log-loss 0.196 with offsets (ridge lambda 0.5) vs 0.2035 with none, constant 0.296. So offsets generalise a little; I used lambda 0.5 (weakest tested, best). Offsets range -1.7 (Madrid) to +1.0 (Austin); they are sizeable, so use them. Tracks with no entry (Las Vegas, Qatar, Abu Dhabi): offset 0.

pytest: `backend/tests/test_race_interaction_params.py` 5 passed. I did not run the full suite or `validate_replay.py` (no model code changed).

## Interface / contract changes
JSON schema as specified, plus extra keys (nothing renamed). How to use:
- Dirty air: if the end-of-lap gap to the car ahead is `< dirty_air.window_s`, add `time_loss_s` to that car's lap. Use `extra_deg_frac` (0.0) for wear; don't invent extra wear. `dirty_air.loss_by_gap_bin_s` lets you taper the loss with gap if you want (about 0.4 at <0.5 s, 0.2 at 0.5-1 s, 0 beyond 1.2 s).
- Overtake: for a car with gap `<= overtake.window_s` (1.5 s) to the car ahead, p = sigmoid(intercept + pace_delta_coef*pace_delta + tyre_age_delta_coef*tyre_age_delta + gap_coef*gap + track_offset.get(track, 0)). `pace_delta` = defender pace minus attacker pace in s/lap (positive = attacker faster), **tyre- and fuel-normalised** (not including the tyre-age effect, which has its own term). It was fit clipped to +-2 s/lap, so clip in the sim. `tyre_age_delta` = defender age minus attacker age. `gap` is the end-of-lap gap in seconds. p is per lap; at gap 0.3 s the observed rate is 55%, at 0.6-0.9 s about 7%, above 1.2 s under 1%.
- Pass cost: on a lap where a pass happens, attacker `+0.12 s` and defender `+1.05 s` on top of the lap's dirty-air term (the defender figure includes being re-boxed behind the attacker plus defending).
- Lap 1: new position = grid position + N(mean, sd) per bin, then re-rank (positions must stay a permutation). Pit-lane starters excluded from the fit.
- `min_follow_gap_s`: floor for a following car's gap after a pass/pit rejoin.
- Extra keys: `overtake.window_s`, `overtake.coef_ci95`, `overtake.base_rate_per_lap`, `dirty_air.*_raw/ci95`, `pass_time_cost_s.*_se`, `lap1.n_by_grid_bin`, `validation`.

## Requests for other roles
- backend-engineer: build the sim's pace_delta from qualifying/practice pace, and note the fit used hindsight race pace (see risks). Passes in the sim should be probabilistic per lap using the formula above; don't clip p except to [0, 1].

## Open questions / risks
- The pace input was estimated from the same race (driver fixed effect from clean-air laps), so the 1.31/s coefficient is optimistic if the sim's pace comes from noisier qualifying. Expect the effective coefficient to shrink; calibrate against the sim's own finishing-order spread (T-034 scorecard).
- Gap is measured only at the line, so true following distances are smaller than recorded and the dirty-air loss may be slightly overstated (the car that is catching a slow car is "dirty" partly because it was already slow). Treat 0.31 s as an upper-ish estimate; the 1.5-2.0 s bins show no loss.
- Overtake mode was on for essentially all laps studied; I did not split by `overtake_enabled`.
- Lap-1 means in the front bins are mildly positive and the back bins negative (regression to the mean plus DNFs and starts from the pit lane); samples are small (n 54-131 per bin).
- Defender pass cost of ~1 s is large and its sd is not in the JSON; if it hurts the finishing-order calibration, reduce it.
