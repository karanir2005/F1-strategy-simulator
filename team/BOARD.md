# Task board

Owned by the orchestrator. Agents read this file and do not edit it. Status: `todo` → `in-progress` → `review` (with QA) → `done` | `blocked`.

## Phase 0: Foundations

| ID | Owner | Status | Depends on | Task |
|---|---|---|---|---|
| T-001 | researcher | done | — | FastF1 schema + 2026 rules brief |
| T-002 | data-scientist | done (QA PASS WITH NOTES; Monaco results refreshed, CLAUDE.md updated) | T-001 | Stint detection fix + richer lap extraction + re-pull |
| T-003 | data-scientist | done (QA via T-005; quote data/scorecard/latest.json) | T-002 | Validation scorecard v2 |
| T-004 | backend-engineer | done | — | `start_age` stints, common random numbers, vectorized Monte Carlo (QA: PASS WITH NOTES → T-007, T-002) |
| T-005 | qa-engineer | done (PASS WITH NOTES; Phase 0 exit met) | T-002, T-003, T-004 | Verify Phase 0 |
| T-006 | frontend-engineer | done | T-004 | Stint input supports used sets (start age) (QA: PASS WITH NOTES → F-1/F-4 into T-008, F-2/F-3 into T-007) |
| T-007 | backend-engineer | done | T-004 | API hardening + paired delta (QA: PASS WITH NOTES, 2 low notes → backlog) |
| T-008 | frontend-engineer | done | T-007, T-006 | Paired delta as primary comparison + T-006 QA fixes (QA: PASS WITH NOTES → backlog) |
| T-009 | data-scientist | done (QA PASS WITH NOTES; 41 races in data/staging/, promotion = T-015) | T-002 | Expand race set via OpenF1 + Jolpica: all 2026 races to date + all 2025 |
| T-010 | researcher | done | — | US GP 2026 brief: COTA, 2025→2026 transferability, what predicts results |
| T-011 | data-scientist | done | T-010 (Jolpica; not blocked on T-009) | Race-outcome predictor (win/podium probabilities), walk-forward validated |
| T-012 | qa-engineer | done (PASS WITH NOTES; findings 1-2 fixed) | T-011 | Verify predictor: leakage, calibration, baselines |
| T-013 | data-scientist | todo | T-011, US GP qualifying | Final US GP prediction with the real grid (2026-10-24) |
| T-014 | data-scientist | done (term built, off by default: no gain on 2025; orchestrator-checked) | T-011 | Per-track "grid stickiness" (overtaking difficulty) term in the predictor, validated quali-to-result on 2026 races |
| T-015 | data-scientist | done (no QA, lean mode) | T-009 | Promote safe 2026 staging races (Barcelona, Madrid, Italy, Azerbaijan, Singapore, Netherlands) to data/processed; add era filter to tire_model; refit, rerun validate_replay + scorecard; update pinned fit-number tests |
| T-016 | data-scientist | done (QA in T-019) | T-009 | Per-track params file (laps, pit loss, base pace, compounds, stint caps) — see team/contracts/track-optimizer.md |
| T-017 | backend-engineer | done (follow-ups applied; QA in T-019) | contract | GET /tracks, POST /optimize (optimal stint plan), track_id on /simulate |
| T-018 | frontend-engineer | done (lint+build clean; browser check in T-019) | T-016, T-017 | Track dropdown + "Find optimal strategy" view |
| T-019 | qa-engineer | done (PASS WITH NOTES; see T-020/T-021) | T-016..T-018 | QA track params, optimizer, UI |
| T-020 | data-scientist | done (no QA; plans partly realistic, see handoff) | T-019 | Make tyre fit physical (SOFT>=MEDIUM>=HARD wear, sensible compound pace offsets) so optimizer plans match what teams really run; rerun scorecard + validate_replay; refresh pinned numbers |
| T-021 | backend-engineer | done (frontend must show new params_used fields) | T-019 | Optimizer: stop 5-lap first-stint artefact (min stint / cap floor interaction), show effective caps in response |
| T-022 | researcher | done | none | UI inspiration brief: what top-tier motorsport/data/product sites do (layout, type, colour, motion, charts) + framer-motion patterns, as an actionable design spec |
| T-023 | backend-engineer | done (QA in T-025) | contract | GET /predict/next and /predict/history per team/contracts/prediction-api.md |
| T-024 | frontend-engineer | done (QA in T-025) | T-022, T-023 | Redesign: shared layout, /predict page, /strategy page, framer-motion, charts |
| T-025 | qa-engineer | skipped (lean mode) | T-024 | QA UI (lint, build, browser pass) and the prediction API |
| T-026 | backend-engineer | done (no QA) | T-020 | Robust p90 stint caps (typical_stint_laps) + recalibration check; SOFT still preferred, constants unchanged |
| T-027 | backend-engineer | done (no QA; FP stays off) | T-023 | Before quali: rehearse grid-mode flow (grid.csv incl. pit lane) across CLI/API/UI; grid-penalty override + page note; test whether `--fp` practice pace helps on 2026 walk-forward (decide on/off) |
| T-028 | data-scientist | done (no QA; per-track wear opt-in, lap noise 0.8->0.6) | T-020 | Per-track tyre wear (replace global curves, shrink to global) + lap-noise (MAD) and pit-loss scripts read all 16 races |
| T-029 | backend-engineer | todo | T-027 | Predictor: revisit track stickiness (`--stick-gamma`) and add per-track safety-car/DNF rates; keep only if walk-forward improves |
| T-030 | backend-engineer | todo | T-028 | Safety cars (and VSC) in the single-car simulator + optimizer |
| T-031 | frontend-engineer | done (no QA; 'what changed' strip needs backend p_win_prequali, folded into T-029) | T-027 | Race-week behaviour on /predict (countdown, grid-set state, what changed since pre-quali, driver detail); phone-width browser check |
| T-032 | backend-engineer | todo | T-031 | Deploy: Vercel frontend + Railway/Render backend, env/CORS config |
| T-033 | frontend-engineer | backlog (later, user) | T-024 | Copy an optimizer plan into the manual comparison with one click |
| T-034 | backend-engineer | backlog (after v1.0.0, user) | T-030 | Phase 1 full-grid multi-car race simulation: the 'final polishing' step that would let the winner prediction beat grid order |

### T-001: FastF1 schema + 2026 rules brief
Confirm against the local `cache/` / a real 2026 session, not just the docs:
- `session.laps` columns needed for Phase 0/1: `Stint`, `FreshTyre`, `Position`, `Time`, `LapStartTime`, `Sector1-3Time`, `Team`, `TrackStatus` code meanings (which code means SC, VSC, red flag, yellow), `IsAccurate`, `Deleted`.
- `session.results` (grid, classification, status/DNF) and `session.weather_data` fields.
- How to compute gap-to-car-ahead per lap reliably.
- 2026 regs relevant to the sim: overtake mode vs DRS, tyre compound/allocation rules, mandatory-compound rule, pit lane speed limits, anything else that changes strategy.

**Done when:** `team/research/T-001-*.md` exists with sourced facts and a recommended column list for T-002.

### T-002: Stint detection fix + richer extraction
- Stint boundaries from `Stint`/pit-in, not `tire_age == 1`. Fix `assign_stint_ids`, `clean_laps` (drop out-laps on used sets too) and `validate_replay.reconstruct_strategy`.
- Add the columns T-001 recommends to `extract_laps_df`. Extract results + weather to sidecar CSVs. Re-pull the 10 races.
- Re-fit the tire model, then report before/after degradation coefficients and `validate_replay.py` numbers.
- If the tire model dict or `Stint` needs a `start_age`, coordinate via handoff (the engine side is T-004).

- **From T-004:** `reconstruct_strategy` must set `Stint.start_age` from the real tyre age at the start of each stint (see `team/handoffs/T-004-backend-engineer.md`).
- **From T-004 QA (N-3):** the fitted SOFT curve is flat (0.00 s from age 2 to 37, +0.05 at age 1), so a used SOFT set simulates as fast as a new one and `start_age` has no effect on SOFT. Find out why (likely the used-set out-lap contamination plus the per-driver demeaning) and fix or document it.

**Done when:** 0 merged used-tyre stints, the re-pull is complete, before/after metrics are in the handoff, and the SOFT curve increases with age (or there's a documented reason it can't be fitted yet).

### T-003: Validation scorecard v2
Script (e.g. `validate_scorecard.py`) that reports:
- leave-one-race-out fitting (model never sees the race it validates)
- per-stint mean s/lap error
- finishing-order rank correlation vs actual, against a grid-order baseline. Before Phase 1 this can be a stub, recorded as N/A.
- strategy-ranking accuracy: for pairs of similar-pace drivers on different strategies, how often the sim predicts who finished ahead
- a `base_laptime` that doesn't use the validated race's own fastest lap

**Done when:** the scorecard runs end-to-end and its baseline numbers are recorded in the handoff.

### T-004: Engine foundations
- `Stint.start_age` (default 1), consumed by `simulate_race`.
- `compare_strategies` uses common random numbers. Same seed gives the same draws for every strategy in an iteration.
- Vectorize across simulations (numpy). Report the timing of 10k × 3 strategies before → after.
- API: add `start_age` to `StintIn` (optional, default 1). Document the contract in the handoff.

**Done when:** pytest passes, results with noise off are unchanged, and the speedup is measured.

### T-005: QA, Phase 0
Verify T-002/3/4 per `.claude/agents/qa-engineer.md`. Specifically: tests for used-tyre stint detection, `start_age` > 1, common random number determinism, and leave-one-race-out leakage.

### T-006: Frontend start-age input
Optional "used set, age N" per stint row, matching T-004's contract. Lint and build must pass.

### T-007: API hardening + paired delta
From `team/handoffs/T-004-qa.md`:
- **N-2:** reject requests whose strategies have different total laps with a clear 422 error. Under common random numbers the shorter race always "wins".
- **N-5:** cap laps per stint and per race in the API to bound O(n_sims × laps) memory.
- **N-1:** add a paired comparison to each result: the per-simulation delta against the best strategy by mean (mean, p10, p90), plus `win_probability` as now. This is the honest headline number under CRN.
- **N-4:** put the CRN / one-car / additive-noise caveats in the module docstring, the `compare_strategies` docstring and the API field descriptions, not only in `simulate_totals`.

Response changes must be additive. Document the exact contract in the handoff for T-008.

### T-008: Paired delta in the UI
Show T-007's paired delta (mean, plus p10–p90) as the primary comparison column. Win probability becomes secondary. Also from `team/handoffs/T-006-qa.md`:
- **F-1:** typing a start age beginning with "1" (e.g. "15") collapses the used-set input mid-typing. Keep the "used set" flag in separate state from `startAge`.
- **F-4:** a cleared numeric field shows "0".
- Mirror T-007's `start_age` upper bound in the input.

### T-009: Expand race set
User decisions (2026-10-19):
- **Barcelona:** in 2026 "Spanish GP" means Madrid and Barcelona is the "Barcelona Grand Prix". Pull Barcelona (round 7), and Madrid too, since all races are wanted. Fix the label and comment in `pull_races.py`.
- **Races:** pull every 2026 race held so far plus all of 2025. 2025 had different regulations, so tag every row with its season. See ROADMAP "Using 2025 data".
- **Sessions:** race and qualifying (grid and quali pace are core predictor features).

The orchestrator is pre-warming the FastF1 cache for 2025 + 2026 R/Q sessions (`laps`, `weather`, `messages`; no telemetry), so extraction should be mostly cache hits. Use the T-002 extractor. Outputs: per-race lap CSVs, plus results/quali sidecars carrying `season`.
### Backlog (small, next backend pass)
- (T-002) `calibrate_lap_noise.py` now gives 1.03 s, but `monte_carlo.DEFAULT_LAP_TIME_NOISE_STD` is 0.8 s. The data scientist should re-run and confirm the value, then the backend engineer updates the constant.
- (T-002) `CLAUDE.md` validation numbers and Spain mention are stale. Replacement text is in `team/handoffs/T-002-data-scientist.md`; the orchestrator applies it after QA.
- `win_probability` exact ties: split evenly between tied strategies instead of crediting the first one listed (orchestrator decision, 2026-10-19; raised in T-004 QA, deferred in T-007).
- `start_age + laps - 1` running past the tyre model's reliable range: that extrapolation needs fixing in the tyre model (data scientist), not with an API cap. See T-007 handoff and T-006 QA F-3.

## Prediction track (deadline 2026-10-24)

### T-010: US GP 2026 brief (researcher)
- **Weekend:** session times and the date each data source becomes available (FP long runs, quali).
- **COTA:** track character (degradation, overtaking difficulty, typical SC rate, pit loss), and the 2025 US GP result and strategies (it was a sprint weekend).
- **2025 → 2026 transfer:** what carries over (driver skill, track-specific effects) and what doesn't (car pace order). Cite the pace shifts.
- **What predicts F1 results:** grid position vs finish correlation (overall and at COTA), the value of practice long-run pace, DNF rates, and published approaches to podium prediction (with sources).
- **Calendar check:** FastF1's 2026 schedule lists round 16 as "Bahrain Grand Prix" at "Kuala Lumpur". Confirm what actually happened.

### T-011: Race-outcome predictor (data scientist)
- **Output:** probability of each driver winning and finishing on the podium (also P(top 10)) for a given race, given the information available before it. One version works pre-quali, another with the real grid.
- **Baselines it must beat:** grid order, championship order, recent-form average.
- **Validation:** walk-forward over 2026 races only. Every prediction uses only earlier races. Report the log-loss/Brier score of win and podium probabilities against the baselines, plus a calibration check.
- **Model:** start simple and interpretable (e.g. ranking/ordered model or XGBoost per CLAUDE.md, with few features). 2025 is allowed only as a down-weighted prior, and it must beat 2026-only in walk-forward to stay in.
- **Deliverable:** a module + script that prints the US GP prediction table.

### T-012: QA the predictor
Leakage (no future races, no post-race info such as the final classification in the features), baseline comparison reproduced, sensible probabilities (sum to 1 for win, 3 for podium).

### T-013: Final US GP prediction
Re-run with the real 2026-10-24 grid (and any penalties). Publish the prediction table.
- (T-007 QA note 1) The memory comment in `simulation.py` ignores the pit-noise matrix. A 99-stop request peaks around 200 MB / 0.27s, which is acceptable. Fix the comment and consider an optional `MAX_STOPS` cap.
- (T-008 QA, frontend) The error box adds "Is the backend running?" even on a 422. A tiny negative delta renders as "-0.00s". The Race Settings input row doesn't wrap at narrow widths, and the table is 8 columns. There are no frontend tests. **Browser check still outstanding for T-006/T-008** (narrow width, dark mode, real typing).

### T-014: Track overtaking-difficulty term (data scientist)
User idea (2026-10-19): use each 2026 race's qualifying to predict the result, compare with the real result, and improve the model. Walk-forward grid mode already does the predict-and-score loop; what's missing is that the grid-vs-form weight is the same at every track (Monaco should stick to quali order, Monza less so). Add a per-track stickiness term, fitted from track history (2025 and earlier is fine here, since overtaking difficulty depends on the track) and shrunk toward the global value. Keep it only if walk-forward 2026 metrics improve; report per-race errors so the biggest misses are visible.

## DATA ACCESS INCIDENT (2026-10-19)

Since about 2026-10-15, `livetiming.formula1.com/static/*` returns **401** for every season (FastF1 issue #978). F1 now requires an F1 TV Access subscription. FastF1's mirror has no 2023–2026 data. `fastf1==3.9.0a0` (alpha, released 2026-10-18) adds authentication. FastF1 does **not** raise on a 401; it warns and returns empty data, so always check `len(session.laps) > 0`.
- **Still available:** the warm `cache/` (10 races from 2026: Australia, China, Japan, Miami, Canada, Monaco, Austria, Britain, Belgium, Hungary), now irreplaceable without a subscription. **Do not delete or overwrite.**
- **Jolpica (free):** results, qualifying/grid, per-lap time and position, pit stops (lap and duration) and standings for 2025 and 2026. No compounds, tyre age, track status or weather.
- **Resolution (2026-10-19):** the user has no F1 TV subscription. **OpenF1** (`api.openf1.org/v1`, free, no auth) still serves historical 2025/2026 data, verified on the 2026 Singapore race: `stints` (compound, `tyre_age_at_start`), `laps` (durations, sectors), `race_control` (SC/VSC/flags), `weather`, `pit` (lane duration) and `intervals` (gaps). T-009 rebuilds the data expansion on OpenF1 + Jolpica.
- **T-011** can run on Jolpica.
- The empty `cache/2025/*` and `cache/2026/*` session directories created at 15:05–15:06 are from a failed warm-up and contain no data.
