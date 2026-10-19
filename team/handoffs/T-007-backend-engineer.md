# T-007 — backend-engineer — API hardening + paired delta

**Status:** done
**Next owner:** qa-engineer: verify T-007 (N-1, N-2, N-4, N-5 from T-004 QA, plus the start_age cap from T-006 QA F-2/F-3). frontend-engineer can start T-008 against the contract below.

## What I did
- **N-2 (unequal total laps).** `compare_strategies` now raises `ValueError` when the strategies' total laps (sum of stint laps) differ. Under CRN the shorter race shares the longer one's first-N lap draws and skips the rest, so it "wins" every iteration. `run_monte_carlo` (single strategy) and `simulate_totals` (the low-level primitive, which QA tests use with different lengths) are unaffected. The API checks the same thing in a `SimulateRequest` validator, so the client gets a 422 that names each strategy's total.
- **N-5 (bounds).** New API constants in `simulation.py`, each documented in a comment:
  - `MAX_RACE_LAPS = 100`: cap on `StintIn.laps` (`le`) and on each strategy's total. The longest race in `data/processed` is Monaco at 78 laps, and the minimum race distance keeps every real race under ~80. Worst case for the noise matrices is 50k x 100 x 8 B = 40 MB each.
  - `MAX_STRATEGIES = 20`: `max_length` on `strategies`. It bounds the (n_strategies x n_sims) arrays, and I added it alongside N-5.
- **start_age cap (orchestrator addition, T-006 QA F-2/F-3).** `StintIn.start_age` is now `le=25` (`MAX_START_AGE`).
  - Real 2026 used sets start at age 2-9 (T-001 research), so 25 leaves about 2.5x headroom for what-ifs.
  - It keeps a used stint's first lap inside the fitted age range: `max_reliable_age` in the current fit is SOFT 37, MEDIUM 51 (cliff 39) and HARD 48.
  - I did **not** add a `start_age + laps - 1` cap. A fresh long stint already runs past `max_reliable_age` (a 50-lap SOFT stint, for example), so a final-age cap wouldn't fix F-3. It would only reject legitimate long stints with a model-dependent number. Extrapolation past the reliable age belongs to the tire model (data-scientist). With both caps, final age is bounded at 124.
- **N-1 (paired delta).** The reference strategy is the one with the lowest mean total (ties go to the first listed, the same rule as `win_probability`). For each strategy, `delta[i] = total[i] - reference_total[i]` within the same simulation, with CRN pairing kept. The result reports the mean, p10 and p90 of that array. The reference gets exactly 0. `compare_strategies` returns `delta_mean`, `delta_p10`, `delta_p90` and `delta_reference` per strategy. The API lifts the reference name to a top-level `reference_strategy` field.
- **N-4 (caveats).** The one-car-in-isolation, additive-noise and CRN caveats are now in three places:
  - the `monte_carlo` module docstring (new CAVEATS block)
  - the `compare_strategies` docstring
  - the pydantic `Field(description=...)` for `win_probability`, `delta_mean`, `delta_p10`, `delta_p90` and `reference_strategy` (visible in `/docs` and OpenAPI)

  I also added unit descriptions to `mean`, `std`, `p10`, `p50` and `p90`.

## Files changed
- backend/app/core/monte_carlo.py: CAVEATS in the module docstring, the equal-laps `ValueError` and the paired delta in `compare_strategies`, docstring rewrite
- backend/app/api/routes/simulation.py: the `MAX_RACE_LAPS`, `MAX_START_AGE` and `MAX_STRATEGIES` caps, total-laps validators (bounded, equal), new response fields, field descriptions
- backend/tests/test_monte_carlo.py: +7 tests
  - unequal laps raise
  - single-strategy and primitive paths are still permissive
  - reference = best mean, with zero self-delta and `delta_mean` == difference of means
  - noise-off delta == deterministic gap
  - same-length, same-stop delta is constant under heavy noise
  - 1-stop vs 2-stop delta p10..p90 spread == 2*1.2816*sigma_pit (only the unshared draw)
  - tie goes to the first listed
- backend/tests/test_api_simulation.py: +8 tests
  - unequal laps 422 with message
  - single strategy has no constraint
  - stint laps cap (100 ok, 101 → 422)
  - total laps cap (60+41 → 422)
  - start_age cap (25 ok, 26 and 500 → 422)
  - strategies cap (20 ok, 21 → 422)
  - delta fields + `reference_strategy`, with old fields still present
  - noise-off delta is a point

## Verification
- `pytest` from repo root: 72 passed before, **87 passed** after.
- `validate_replay.py`: **unchanged**. China +26.7s (+0.50%), Britain -150.2s (-2.96%), Austria -40.3s (-0.77%). I didn't touch the lap loop. This was run as a regression check.
- Perf: `compare_strategies`, 10k sims x 3 strategies with the real tire model and seed 42, took **0.0118 s/call** (T-004: 0.013s). The delta computation adds no measurable cost.
- Worst case the API now allows (50k sims, 20 strategies, 100 laps and 19 stops each): 0.16s, tracemalloc peak **128 MB**. Before this task the laps were unbounded and could reach MemoryError.
- Example output with the T-004 strategies, seed 42:
  - M25/H32: win 0.0, delta +0.31 [p10 +0.31, p90 +0.31]
  - S15/M20/H22: win 0.02, delta +9.73 [p10 +3.43, p90 +16.12]
  - H30/M27: reference, win 0.98, delta 0
  - This is exactly the N-1 point. The 0%/98% split is really a 0.31s gap.
- Not verified: live HTTP latency through uvicorn (tested via TestClient only), and the frontend against the new fields (T-008).

## Interface / contract changes

**Python**
- `compare_strategies(...)`: the signature is unchanged.
  - It raises `ValueError` if the total laps differ across strategies.
  - Each per-strategy dict gains `delta_mean: float`, `delta_p10: float`, `delta_p90: float` (seconds) and `delta_reference: str`.
- `run_monte_carlo`, `simulate_totals` and `simulate_race` are unchanged.

**HTTP `POST /simulate` request.** These are new bounds only, with no new fields. Each violation returns 422.
```
SimulateRequest {
  base_laptime: float            // seconds, > 0
  strategies: StrategyIn[]       // 1..20 (NEW max 20), names unique,
                                 // NEW: every strategy's total laps (sum of stint laps) must be equal
                                 // NEW: every strategy's total laps must be <= 100
  n_simulations?: int            // default 10000, 1..50000
  lap_time_noise_std?: float     // seconds, >= 0, default 0.8
  pit_loss_noise_std?: float     // seconds, >= 0, default 5.0
  pit_loss_seconds?: float       // seconds, >= 0, default 23.5
  seed?: int | null
}
StrategyIn { name: string; stints: StintIn[] /* >= 1 */ }
StintIn {
  compound: "SOFT" | "MEDIUM" | "HARD"
  laps: int                      // 1..100 (NEW max 100)
  start_age?: int                // 1..25 (NEW max 25), default 1 (= new set); must be omitted, not null
}
```
Example 422 bodies, using FastAPI's standard `detail` list. `api.ts` already stringifies `detail`.
- Unequal totals: `{"detail":[{"type":"value_error","loc":["body","strategies"],"msg":"Value error, all strategies must cover the same total race laps (sum of stint laps) to be compared, got {'A': 40, 'B': 30}", ...}]}`
- Total too long: `msg` = `"Value error, total race laps (sum of stint laps) must be at most 100, got {'A': 101}"`, `loc` = `["body","strategies"]`
- start_age 26: `{"type":"less_than_equal","loc":["body","strategies",0,"stints",0,"start_age"],"msg":"Input should be less than or equal to 25","ctx":{"le":25}}`
- laps 101: the same shape with `loc [..., "laps"]` and `ctx {"le":100}`. 21 strategies give `type "too_long"` at `loc ["body","strategies"]`.

**HTTP `POST /simulate` response.** The changes are additive, and no field was removed or renamed.
```
SimulateResponse {
  reference_strategy: string     // NEW. name of the strategy all delta_* are measured against:
                                 //      lowest mean total time; ties -> first listed in the request
  results: StrategyResult[]      // same order as request.strategies (unchanged)
}
StrategyResult {
  name: string
  mean, std, p10, p50, p90: float   // seconds, total race time (unchanged)
  win_probability: float            // 0..1 (unchanged semantics; near-0/1 under CRN, see caveats)
  delta_mean: float                 // NEW. seconds. mean over sims of (this total - reference total, same sim).
                                    //      >= 0 for every strategy (reference is best by mean); exactly 0 for the reference
  delta_p10: float                  // NEW. seconds. 10th percentile of that paired gap; 0 for the reference
  delta_p90: float                  // NEW. seconds. 90th percentile of that paired gap; 0 for the reference
  lap_times: float[]                // unchanged (noise-free trace)
  pit_laps: int[]                   // unchanged
}
```
Notes for T-008:
- `delta_mean >= 0` always holds, because the reference has the lowest mean. `delta_p10` can be **negative** for a non-reference strategy, meaning the strategy beats the reference in at least 10% of sims (through pit-noise draws).
- `delta_p10 == delta_p90` is common. Strategies with the same stop count share every draw, so the paired gap is a single number, which is correct under CRN. Show it as one value rather than a degenerate range.
- Model error is tens of seconds (replay errors 27-150s). The QA suggestion to label sub-second deltas "no meaningful difference" is the frontend's call. I did not add a threshold field.

## Requests for other roles
- frontend-engineer (T-008):
  - Build against the response contract above.
  - Mirror the request bounds client-side: laps 1..100 per stint, total <= 100, equal totals across strategies, start_age 1..25 and at most 20 strategies. That gives inline errors instead of a 422.
  - Don't edit backend files.
- data-scientist (T-002, existing): T-006 QA F-3 still stands for long stints. `predict_delta` extrapolates linearly past `max_reliable_age` (SOFT 37). The start_age cap only keeps used-set first laps inside the fitted range.

## Open questions / risks
- QA's N-1 also suggested splitting `win_probability` evenly on exact ties. The current rule is that the first-listed strategy takes 100%. I left it as is because the brief said to keep "`win_probability` as now". The orchestrator can decide.
- The bounds 100 / 25 / 20 are judgment calls with documented rationale, not fitted values. If 2026+ calendars or what-if use cases need more, change the constants in `simulation.py`. They are in one place, and the tests import them.
- `delta_reference` lives in the Python dict (once per strategy, all identical). In the API it is lifted to the top level and stripped from each result.
- `cache/fastf1_http_cache.sqlite` may show as modified after `validate_replay.py`. That comes from FastF1, not from this task.
