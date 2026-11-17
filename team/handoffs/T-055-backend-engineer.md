# T-055 — backend-engineer — Safety-car-reactive pit policy

**Status:** done (with an honest "modest and model-dependent" verdict, see Results)
**Next owner:** orchestrator — decide whether to commit; data-scientist (T-060 or later) for the tyre-cliff / SC-wear placeholders that drive the gain.

## What I did
- `backend/app/core/pit_policy.py` (new). Policy class: "base plan X; if an SC (or VSC) STARTS on lap L in [a, b] and tyre age >= k, box now for compound C; else keep the plan." `PitPolicy(window_lo, window_hi, min_age, compound, trigger)`, trigger = `sc` or `sc_or_vsc`.
  - "Box now" = pull-forward: the plan's next stop moves to the SC lap on C (same number of stops, the new stint runs to where the plan's next stint ended); on the last stint it is an extra stop. Not allowed (policy does not fire) when it leaves one compound, makes the C stint longer than the optimizer's stint cap for C at that track, leaves < 5 laps, or the SC lap is the plan's own stop lap / lap 1.
  - Search: a coarse grid, not DP. Reason: the value of a state needs a full-field race in the grid sim, and the policy class has 4 numbers. The grid is made cheap by an exact shortcut: a window policy's per-run outcome equals the outcome of "fire at the j-th eligible SC/VSC start", so for each (k in 4/8/12/16, C in S/M/H) the grid sim is run once per event slot (j = 0..2, common random numbers) and all windows [a, b] and both triggers are then optimised on the tables with no more simulation (a test checks the shortcut reproduces the direct policy run exactly). About 36 grid-sim runs of 600 sims.
  - Scoring: the full-grid sim with the policy on one hero car (default: the pole sitter on the optimizer's #1 plan); paired against the same car on its fixed plan with the same draws (CRN). The rule is selected on seed s and its gain is reported from a fresh seed s+1, so the interval excludes selection bias. Interval = 95% normal half-width of the paired per-run difference. Also `if_field_follows`: every other car follows the same rule too.
- `race_grid_sim.py`: ONE additive optional parameter `plan_hook=None` (callable(sc, vsc, base_tables) -> {car index: per-run (N, L) tyre/age/pit tables}), called once after the SC timeline is drawn, plus a 10-line branch in the lap loop that swaps those tables in. `plan_hook=None` is bit-identical to before (test). I re-read the file before each edit (T-060 was editing it); my edits are the 4 spots containing `plan_hook` / `overrides`.
- `POST /strategy/policy` (`api/routes/pit_policy.py`, wired in `main.py`), MCP tool `optimize_pit_policy` in `mcp_server.TOOLS` (+ `CAVEATS_POLICY`), schema in `chat_service.TOOL_SCHEMAS`.
- Frontend: `frontend/components/strategy/pit-policy-card.tsx` (button-triggered card under the optimizer results on /strategy, rule sentence + gain with interval + fire rate + "if every other car follows"), `runPitPolicy` and types in `frontend/lib/api.ts`, mounted in `frontend/app/strategy/page.tsx`. I touched frontend files because the task asked for it.
- `validate_pit_policy.py` (new, repo root) -> `data/scorecard/pit_policy_validation.json`.

## Files changed
New: `backend/app/core/pit_policy.py`, `backend/app/api/routes/pit_policy.py`, `backend/tests/test_t055_pit_policy.py`, `validate_pit_policy.py`, `frontend/components/strategy/pit-policy-card.tsx`, `data/scorecard/pit_policy_validation.json`. Edited: `backend/app/core/race_grid_sim.py` (plan_hook only), `backend/app/main.py`, `backend/app/mcp_server.py`, `backend/app/services/chat_service.py`, `frontend/lib/api.ts`, `frontend/app/strategy/page.tsx`.

## Results (model output, 3,000 fresh-seed runs per row, hero on the optimizer #1 plan, 2026 grid/quali)
Gain = expected points vs the same car on the fixed plan, 95% interval; fires = share of runs where the rule fired.

| Track, hero (grid) | Best rule found | Gain pts | Gain places | Fires | Gain when fired | If field follows |
|---|---|---|---|---|---|---|
| Austin, ANT (P1) | SC any lap 2-55, age >= 4, Hards | +0.74 (±0.09) | +0.25 (±0.03) | 24% | +3.0 pts | +0.87 (±0.15) |
| Austin, ALB (P17) | same | +0.04 (±0.02) | +0.25 (±0.03) | 24% | +0.2 | +0.01 |
| Monaco, ANT | SC laps 27-77, age >= 4, Hards | +0.87 (±0.10) | +0.23 (±0.03) | 36% | +2.4 | +1.37 (±0.20) |
| Japan, ANT | SC laps 2-52, age >= 4, Hards | +0.46 (±0.08) | +0.12 (±0.02) | 24% | +1.9 | +0.66 (±0.14) |
| Belgium, ANT | SC laps 20-43, age >= 4, Hards | +0.26 (±0.05) | +0.09 (±0.02) | 16% | +1.7 | +0.21 (±0.07) |

Honest reading:
- The search always picks a full-SC trigger, mostly a wide window and the smallest age threshold (4), box for Hards. That is "any SC after the first stint's early laps, take a fresh hard set". The window is wide because the stint cap and the lap-remaining rule already remove the early/late cases; the thresholds the search can tell apart are few. Do not read the "[2,55]" as a tuned window. The k grid starts at 4, so "age >= 4" is the edge of the grid.
- Where the gain comes from: mostly a mid/late SC while the car is on its last stint (extra stop at the SC-discounted pit loss onto fresh Hards), i.e. the model values a fresh set for the last 15-25 laps above a ~9 s stop. That value rests on the tyre-wear curve and the PLACEHOLDER cliff penalties and on tyres not wearing less under SC. Treat the gain size as model-dependent; the sign (box under a late SC) agrees with F1 practice, the size is unverified.
- The gain is about 0 for a back-marker (ALB: +0.04 pts) because there are few points to move; position gain is the same ~0.25 places. Screen-sample gains are within about 10% of the fresh-seed gains (no strong selection effect).
- "If field follows" is the rule vs the fixed plan with the whole field reacting; it is a best-response check (the hero should react if everyone does), not an equilibrium. It did not shrink the gain, which means the benefit is not just "one car gets a free stop": the tyre value inside the model is carrying it.
- Speed: search + 3,000-run evaluation + 1,500-run field check: 10-18 s per request on this machine (600 screen sims, about 36 slot runs). With `field_sensitivity=false` and n_sims 2,000 it is about 9 s. The MCP/chat tool defaults to the cheaper settings (n_screen 400, n_sims 2,000, no field check).

## Validation vs real 2026 SC/VSC laps (`validate_pit_policy.py`, 18 of 19 pulled races have events; Brazil has none)
For each race: policy from the sim for that track (never sees lap data); real events from per-lap SC/VSC flags (24 VSC and 10 SC start events; the 90 SC pairs are 10 events x ~9 top-10 cars still running); for each top-10 finisher still running at the event start, did the policy fire vs did the driver pit in on the start lap or the next. 330 (driver, event) pairs.
- Agreement 71.2% vs 70.9% for "never box": no better than a silent policy overall. Drivers boxed in 29.1% of pairs (96); the policy fired in 25.2% (83); both 42. Precision P(boxed | fires) 50.6%, recall 43.8%.
- Full SC events (90 pairs): agreement 47.8%, but 55.6% of drivers boxed, so "always box under an SC" would have agreed more (55.6%) than the policy does. VSC events (240 pairs): agreement 80.0%, boxed 19.2%; the policy almost never fires on VSCs (it picked trigger `sc`), so that is the never-box rate.
- Compound match when both boxed: 21% (9 of 42), because the sim always recommends Hards; real teams took the other compounds mostly. This is the clearest disagreement and points at the compound pace offsets (SOFT/MEDIUM/HARD tie on pace in the 2026 fit, so the sim only sees wear).
- Biases: a driver already due to stop that lap counts as "boxed" (the data has no planned-stop lap); the policy used here is the pole sitter's plan/policy for the track, applied to all ten drivers; per-race tyre ages come from the real laps. So agreement is a coarse check, not a score of the policy. The sim and real teams agree on the sign for late SCs more than for early ones; I did not slice by lap (330 pairs, small cells).

## Verification
- Full `pytest` from repo root: **418 passed** (178 s) after my changes (includes T-060's concurrent edits as they stood). New file `backend/tests/test_t055_pit_policy.py`, 13 tests: fires only on an SC (or SC/VSC) start inside the window (checked against the masks), only in the window, only when age is met; determinism with a seed (evaluation, search, API); gain exactly 0 and never fires with `safety_cars` off; runs where it does not fire are identical to the base run (CRN); the slot-table shortcut equals the direct run; empty hook = no hook; plan rewriting rules; API smoke, 404/422, no-SC response; MCP + chat tool registration.
- `npx tsc --noEmit` clean, `npm run lint` clean. I did not run `npm run build` and did not view the card in a browser.
- `validate_replay.py` not run (race_simulator.py untouched). The grid-sim validation (`validate_grid_sim.py`) not re-run; `plan_hook=None` is bit-identical (test).

## Interface / contract changes
`POST /strategy/policy` request: `{track_id: str, driver?: str (default pole sitter), round?: int, base_plan?: [{compound: SOFT|MEDIUM|HARD, laps: int, start_age?: int}] (default optimizer #1; must sum to race laps), n_sims: int=3000 (200..10000), n_screen: int=600 (100..3000), seed: int=0, safety_cars: bool=true, field_sensitivity: bool=true, total_laps?: int}`. 404 unknown track, 422 unknown driver / bad plan / no data.
Response: `{track_id, round, total_laps, driver, team, grid, base_plan: [{compound, laps, start_age}], policy: {window_lo, window_hi, min_age (laps), compound, trigger: "sc"|"sc_or_vsc"} | null, rule: str, significant: bool (95% interval of the points gain above 0), gain: {points, points_ci95, positions, positions_ci95 (places gained, + = better), fire_rate (0..1)} | null, points_gain_when_fired, points_gain_when_fired_ci95, positions_gain_when_fired, share_of_fires_better, share_of_fires_worse, fire_lap_mean, base_exp_points, base_mean_position (all nullable), if_field_follows: {same shape as gain} | null, screen_gain_points, n_sims, n_screen, seed, notes: [str], assumptions: [str]}`. `policy`/`gain` are null when no event could trigger an allowed stop (e.g. `safety_cars=false`). When `significant` is false `rule` starts "No safety-car rule beats the fixed plan by more than the simulation noise. Best candidate: ...".
Python: `pit_policy.PitPolicy`, `PolicyContext`, `build_context(track_id, tire_model, driver, base_plan, rnd, grid, safety_cars, seed, total_laps)`, `search_policy(ctx, n_screen, n_eval, seed, field_sensitivity, n_field)`, `evaluate_policy(ctx, policy, n_sims, seed, field_reacts)`; `simulate_grid(..., plan_hook=None)`. MCP/chat tool: `optimize_pit_policy(track_id, driver?, base_plan?, n_sims=2000, n_screen=400, seed=0, field_sensitivity=false)` returning the response above with rounded numbers, `base_plan` as strings like "M24", and `caveats`.

## Known simplifications
One hero car with one rule, others fixed (plus the all-follow check); new tyre goes on at the start of the SC lap (one lap optimistic); wear and fuel not reduced under SC (engine-wide); a stop under a VSC is priced with the fitted vsc_pit_loss_ratio but VSC rarely wins in the search; max 3 events per run in the search; no reaction to position, gaps or the cars behind (a rule on "gap to the car behind" is the obvious next policy feature); stint caps come from the optimizer's p90 rule, so plans longer than observed are never created; the rule cannot choose a two-stop-to-one-stop change (pull-forward keeps stop count except on the last stint); lap-1 SC (Belgium, Italy) cannot be reacted to by construction.

## Requests for other roles
- data-scientist: (1) SC/VSC tyre-wear and fuel effect (laps under SC age tyres less); (2) the compound pace offsets and cliff placeholders: the policy always recommends Hards and the real data disagree on compound in 79% of the matching stops, and the gain size depends on these; (3) SC length by lap / track so the window can be tied to real events.
- frontend: card is minimal and untested in a browser; the card has no driver picker (default pole sitter), the API accepts `driver`.

## Open questions / risks
- Is "box under every full SC after lap ~20" what teams do? The data say about 56% of top-10 cars box under a full SC (90 pairs), so the rule is directionally plausible but the compound it recommends is not.
- The search is on one hero; gain varies a lot by car (ALB ~0). The route does not average over the field.
- The card runs on demand because a request takes 10-20 s; a cached/precomputed per-track table would be needed to show it automatically.

## T-061 re-run (on the T-060 calibrated grid sim; the numbers above are the pre-calibration ones)
Same code and settings (600 screen sims, 3,000 fresh-seed runs, 1,500 for the field check). New defaults: pass rate 22/race instead of 60, pace spread about 2x wider, looser SC bunching, pit-stop jitter 12 laps.

| Track, hero (grid) | Best rule found | Gain pts | Gain places | Fires | Gain when fired | If field follows |
|---|---|---|---|---|---|---|
| Austin, ANT (P1) | SC laps 2-50, age >= 4, Hards | +0.22 (±0.05) | +0.08 (±0.02) | 23% | +1.0 | +0.24 (±0.08) |
| Austin, ALB (P17) | SC/VSC laps 2-55, age >= 4, Hards | -0.00 (±0.01), NOT significant | -0.04 (±0.02) | 48% | 0.0 | -0.02 |
| Monaco, ANT | SC laps 27-77, age >= 8, Hards | +0.46 (±0.08) | +0.14 (±0.03) | 33% | +1.4 | +0.59 (±0.13) |
| Japan, ANT | SC laps 2-45, age >= 8, Hards | +0.17 (±0.05) | +0.05 (±0.02) | 17% | +1.0 | +0.14 (±0.08) |
| Belgium, ANT | SC/VSC laps 2-31, age >= 12, Hards | +0.26 (±0.07) | +0.09 (±0.02) | 12% | +2.2 | +0.26 (±0.09) |

Reading: every gain for the pole car is roughly halved or more (Austin 0.74 -> 0.22, Japan 0.46 -> 0.17, Monaco 0.87 -> 0.46), the back-marker gain is now zero (the search's rule is not significant, so the API returns the "no reliable rule" wording). The shape is unchanged (late-ish SC, fresh Hards) but the thresholds moved (age 8 or 12 more often) and Belgium's best window is now early. With wider pace spread the car has fewer position fights to win with a fresh set, and more random pit diversity in the field means fewer cars are left to be passed. Still model-dependent on the placeholder tyre cliff.

Validation vs real 2026 SC/VSC laps (`validate_pit_policy.py`, same 330 pairs, 18 races; `data/scorecard/pit_policy_validation.json` overwritten):
- Agreement 70.6% vs 70.9% for never-box (was 71.2% vs 70.9%): still equal to the silent baseline, within noise. Fires 27.6% of pairs (91), boxed 29.1% (96), both 45. Precision 49.5%, recall 46.9%.
- Full SC (90 pairs): agreement 58.9% (was 47.8%) vs 55.6% always-box/base rate; VSC (240): 75.0% vs 80.8% never-box (the policy now sometimes fires on VSC).
- Compound match when both boxed: 35.6% of 45 (was 21%), still low: the sim recommends Hards, teams mostly did not.
Conclusion unchanged: the rule is a model finding about the engine's tyre values, with no demonstrated agreement with real team behaviour beyond the never-box rate. The pit-policy card, API `assumptions` and the MCP caveats now say this explicitly.

T-061 text changes: `grid_sim.py` ASSUMPTIONS and `mcp_server.CAVEATS_GRID` no longer say "judgement placeholders" (they cite the T-060 fits, the thin-evidence constants and the finishing-order result); I did not add `calibration_source` to the `/simulate/grid` response (contract unchanged).
