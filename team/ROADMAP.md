# Roadmap: from strategy calculator to strategy simulator

Written 2026-10-19. Extends the build order in `CLAUDE.md`.

## Why the current engine can't rank strategies

1. **The validation can't separate strategies.** 0.5–3% error on a ~5,400s race is 27–160s. Real strategy differences are 5–15s. `validate_replay.py` also uses the driver's own fastest lap from the race being validated as `base_laptime`, which leaks the answer into the check.
2. **It simulates one car alone.** No undercut/overcut, traffic or track position. Without these the best strategy is just minimum degradation + pit loss, which you could work out on paper.
3. **The Monte Carlo noise doesn't change the ranking.** Symmetric iid lap noise averages out, so the ranking equals the noise-free run. Each strategy also gets independent random draws rather than common random numbers.
4. **Bug: used-tyre stints aren't detected.** Per T-001, 57 of 343 real tyre changes (17%) are onto a used set (start age 2–9). Another 44 pit passes keep the same tyre (SC pit-lane passes, drive-throughs). The earlier "29%" figure lumped both together. `assign_stint_ids` and `reconstruct_strategy` only start a new stint when tire_age == 1, so those stints get merged, used-set out-laps get through `clean_laps`, and replays simulate the wrong strategy.
5. **No safety cars and no per-track parameters.** SC laps are filtered out of the fit but never simulated. Degradation, pit loss and compound pace are global.

## End goal and deadline (added 2026-10-19)

**Predict the winner and podium for the 2026 United States GP** (Austin, race Sunday 2026-10-25, conventional weekend, qualifying Saturday 2026-10-24). Las Vegas (2026-11-21) is the next US race after that.

Two tracks run in parallel:
- **Prediction track (deadline 2026-10-24, after qualifying):** a finishing-order model, in effect an early slice of Phase 7. Trained on 2026 form with 2025 as a down-weighted prior, evaluated walk-forward on 2026 races, and updated with the real grid on Saturday. Outputs win and podium probabilities with honest uncertainty.
- **Simulation track:** Phase 0 → Phase 1 as below, aimed at Las Vegas and later. Once the full-grid sim exists, its outputs become features for the predictor.

**Using 2025 data:** 2026 brought new cars, power units, tyres and Overtake Mode, and team pecking orders shifted. Use 2025 for things that depend on the track (SC rates, pit loss, track character, the COTA strategy pattern) and for driver-skill priors. Don't use it for 2026 car pace. Every row carries an `era` / `season` field, and any model mixing eras must show it beats a 2026-only model in walk-forward CV.

## Phases

### Phase 0: Foundations
- Detect stints from FastF1 `Stint`/`FreshTyre` (or a pit-in on the previous lap). Add `start_age` to `Stint`.
- Extract `Position`, `Time`/`LapStartTime` (gaps), `Stint`, `FreshTyre`, `Team`, sector times, raw `TrackStatus` codes (SC vs VSC), grid, classification and `weather_data`.
- **Validation scorecard v2:** leave-one-race-out fitting, per-stint s/lap error, finishing-order correlation, and strategy-ranking accuracy. Every later phase must improve on it.
- Common random numbers across strategies, and numpy-vectorized Monte Carlo.

### Phase 1: Full-grid race simulation (the change that matters most)
- Pace per car/driver from qualifying and practice long runs, shrunk toward the team mean.
- Gaps and dirty air (time loss + extra degradation when within ~1s), fitted from 2026 laps.
- Overtake probability = f(pace delta, tyre-age delta, gap, track), fitted from 2026 data (overtake mode, not DRS).
- Pit rejoin at the real gap, which makes undercut and overcut fall out naturally. Fitted lap-1 position changes.
- Output a finishing-position distribution per car.

### Phase 2: Random events and reactive strategy
- SC/VSC/red-flag rates and durations per track (2018–2025 data is fine here, since SC frequency depends mostly on the track). Field bunching, cheaper stops under SC/VSC, DNFs.
- Strategies become **policies** ("box early if SC after lap 15", "cover a rival within Xs").

### Phase 3: Track, compound and conditions calibration
- Per-track, per-compound degradation and pace with partial pooling. Fixes the inverted SOFT-pace offset once traffic is controlled for.
- Track temperature effect, per-track pit loss (pit-lane time minus on-track equivalent), tyre allocation and the two-compound rule. Retire placeholders.

### Phase 4: Optimizer (DP + GA from CLAUDE.md)
- DP on the deterministic model for good starting candidates, then a GA (DEAP) over policy parameters, scored by Monte Carlo against a field that reacts.
- Objective: expected position/points, with the risk spread reported. Output the top 3 policies, a sensitivity analysis and the pit window.

### Phase 5: Weather
- Rain as a random event, with slick/inter/wet crossover lap times fitted from extra rain races.

### Phase 6: Features
- **What-if race replay** (change one decision in a real race and re-simulate), a pre-race planner, and later a live-timing mode.

### Phase 7: ML predictor
- XGBoost using simulator outputs plus grid/form features, with time-series CV only.

### Alongside the phases
- Data: all 2026 races, plus 2018–2025 for SC and weather statistics.
- API: async jobs, Redis caching of fitted models.
- Frontend: race traces, position histograms, pit-window heatmap.

## Exit criteria

| Phase | Done when |
|---|---|
| 0 | Stints correct; leave-one-race-out per-stint error reported; strategy-ranking baseline recorded |
| 1 | Finishing-order correlation clearly beats a "grid order" baseline |
| 2 | Simulated SC count per race matches history; ranking accuracy improves |
| 3 | Per-track per-stint error down; fitted compound offsets have the right sign |
| 4 | Optimizer's top pick matches what top teams ran, or the gaps are explained |
