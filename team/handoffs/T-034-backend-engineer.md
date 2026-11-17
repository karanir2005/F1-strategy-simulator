# T-034 — backend-engineer — Full-grid multi-car race simulation

**Status:** done
**Next owner:** orchestrator — decide whether to commit. Data scientist (optional): fit the placeholders listed below. Frontend: nothing required, a `/simulate/grid` contract is below if wanted.

## What I did
- `race_grid_sim.py`: numpy-vectorised over N runs x D cars, lap loop. Cumulative race time per car, so pit rejoin at the real gap (undercut/overcut emerge), dirty air, per-lap logistic overtake attempts (T-053 formula, pace delta clipped to +-2), pass costs, held-behind at min_follow_gap, lap-1 shuffle, SC (bunching, slower laps, cheaper stops) and VSC from `safety_cars.py`, DNFs from the predictor's rates, forced SC. Common random numbers: all draws have strategy-independent shapes (a test checks retirements do not move when one car's plan changes). Deterministic by seed.
- `grid_inputs.py`: entry list and grid (real grid if the race ran, else quali classification, or an override), pace from Q1 gaps shrunk to team mean blended with the predictor's earlier-race ratings (quantile-matched), DNF rates from `predictor.dnf_probabilities`, default strategies = optimizer top-3 plans (60/25/15) with +-2 lap stop jitter per car; per-driver overrides.
- Interaction params loaded from `data/sim_params/race_interaction.json` (T-053, real); per-key PLACEHOLDER fallback if missing (`placeholder_keys` reports it). Conventions follow T-053 (lap-1 mean = places lost, gap at start of lap = end of previous lap, overtake window 1.5 s). `extra_deg_frac` is 0 as fitted.
- `POST /simulate/grid` (new `api/routes/grid_sim.py`, wired in `main.py`). `/simulate` and `/optimize` unchanged.
- `forced_sc_lap` / `forced_sc_laps` on the grid sim, and on `monte_carlo.simulate_totals` / `compare_strategies` (default None = bit-identical to before; if `safety_cars` is None the forced SC is the only neutralisation). Helpers `apply_forced_sc`, `default_sc_length` in `safety_cars.py`.
- MCP tool `simulate_grid` in `mcp_server.py` (TOOLS) and schema in `chat_service.TOOL_SCHEMAS`, both with `forced_sc_lap`.

## Files changed
- `backend/app/core/race_grid_sim.py`, `backend/app/core/grid_inputs.py` (new); `backend/app/core/safety_cars.py`, `monte_carlo.py` (forced SC); `backend/app/api/routes/grid_sim.py` (new); `backend/app/main.py` (router); `backend/app/mcp_server.py`, `backend/app/services/chat_service.py` (tool); `validate_grid_sim.py` (new, repo root); tests `test_t034_grid_sim.py` (new, 13), `test_mcp_server.py`, `test_chat.py` (one test each added, tool list extended). Validation output `data/scorecard/grid_sim_validation.json`.

## Verification
- Full `pytest`: 403 passed, 2 failed before my last fix; after it the two (new MCP/chat tests, a `round` argument shadowing the builtin) pass: re-ran the 3 affected files, 35 passed. I did not re-run the whole suite after that last fix. `validate_replay.py` not run (race_simulator.py untouched).
- Perf: 10,000 sims x 22 cars x 56 laps = 3.5 s (Austin, with SC); 1,000 sims 0.3 s.
- Validation (`python validate_grid_sim.py --sweep`), leave-one-race-out: tyre model refit without the held-out race, that race's real grid and Q1 times, pace/DNF from earlier rounds only, 19 rounds (R2-R20), 1,000 sims, ranked by expected points. Spearman vs official finishing position (all drivers, retirements in their official slots):

| Ranker | Spearman | Spearman (classified only) | Top-3 hit rate | Winner hit |
|---|---|---|---|---|
| Grid order | 0.657 | 0.758 | 0.596 | 0.684 |
| Plackett-Luce predictor (walk-forward) | 0.702 | 0.805 | 0.614 | 0.684 |
| **Grid sim (default)** | **0.700** | **0.803** | **0.614** | 0.526 |

  Sim minus grid Spearman: +0.044, se 0.030, better in 10 of 19 races. So: it beats the grid order on rank correlation but within noise (1.5 se), ties the PL predictor on Spearman and top-3, and is WORSE than both at picking the winner (10/19 vs 13/19; the sim spreads win probability away from the pole sitter more than reality does). Honest verdict: no demonstrated gain over the existing predictor; do not wire it into /predict.
  Sensitivity (not tuned, defaults were fixed beforehand): pace scale 0.3/0.6/1.0 with quali weight 0.5 gives Spearman 0.686/0.700/0.704; quali-only pace (weight 1.0) gives 0.690/0.688 and top-3 0.51. The ratings blend is what keeps top-3 up.
- Overtake count: about 60 successful passes per race in the sim (50-95); the T-053 data had about 13 recorded passes per race under its stricter definition. The sim likely over-passes (trains stay inside 1.5 s too often). I did NOT calibrate this away; it is the first thing to look at if the sim is used for strategy.
- Not verified: calibration of the P(win) values themselves (log-loss vs PL), the 2025 races, wet races.

## Interface / contract changes
`POST /simulate/grid` request: `{track_id: str, round?: int, grid?: [{code, grid}] (0 = pit lane), n_sims: int=2000 (1..20000), seed?: int=0, safety_cars: bool=true, forced_sc_lap?: int, forced_sc_laps?: int (default median fitted SC length), strategies?: {CODE: [{compound: SOFT|MEDIUM|HARD, laps: int, start_age?: int}]}, total_laps?: int}`. 404 unknown track, 422 for bad grid/strategy/no data.
Response: `{track_id, round, total_laps, n_sims, seed, grid_source, drivers: [{code, team, grid, pace_delta_s, p_dnf_input, strategy: [{compound, laps, start_age}], p_win, p_podium, p_top10, p_dnf, exp_points, mean_position, p10_time_s, p50_time_s, p90_time_s (seconds, null if never finishes), position_probs: [P(pos 1..n)]}] (sorted by exp_points), mean_overtakes_per_race, mean_sc_laps_per_race, interaction_source, interaction_placeholder_keys, forced_sc_lap, assumptions: [str]}`.
Python: `simulate_grid(cars, base_laptime_s, tire_model, total_laps, n_sims, seed, ..., safety_cars, forced_sc_lap, forced_sc_laps, track_id)`; `compare_strategies(..., forced_sc_lap=None, forced_sc_laps=None)`. MCP/chat tool `simulate_grid(track_id, round?, n_sims=1000, seed=0, safety_cars=true, forced_sc_lap?, forced_sc_laps?, strategies?, top_n=22)`.

## Known simplifications
No lapping/blue flags (positions by cumulative time), no red flags or weather, no pit-lane queues, plans fixed (no reactive stops), one overtake attempt per car per lap, lap 1 is a re-ranking not a physical start, DNF lap uniform and does not trigger an SC, SC/VSC wear and fuel not reduced, tyre-delta and base lap time are the global fits (default `use_track_wear` off), only the track's `track_offset` varies overtaking by circuit (0 for Las Vegas/Qatar/Abu Dhabi). Leaks in validation: `track_params.json` (base lap time, pit loss, stint caps) includes the held-out race. Pace spread per sim is wider than the PL predictor's because of the pace noise terms below.

## Requests for other roles
- data-scientist: fit the PLACEHOLDERS (all labelled in code): `grid_inputs.RACE_PACE_SCALE` 0.6 (race vs Q1 spread), `DRIVER_SHRINK` 0.5, `QUALI_WEIGHT` 0.5, `race_grid_sim.TEAM_PACE_SD_S` 0.20 / `DRIVER_PACE_SD_S` 0.15 (race-day pace uncertainty), `SC_GAP_S` 0.9 and `SC_GAP_CLOSURE` 0.5 (SC bunching), `LAP1_SPACING_S` 0.7, strategy mix `PLAN_WEIGHTS`/`PIT_JITTER_LAPS`; and a pass-count calibration (the sim over-passes, see above) possibly by scaling the overtake intercept. The T-053 pace coefficient was fitted on in-race pace; with quali-based pace it is probably overconfident, which the pass count suggests too.

## Open questions / risks
- Winner hit rate below grid order: a pole bonus (grid-slot term) or less pace noise may be needed; untested to avoid tuning on the 19 races.
- Real `/simulate/grid` default for a race with no quali yet needs an override grid or returns 422/uses the latest quali; Las Vegas (R21) quali is not in the cache yet.
