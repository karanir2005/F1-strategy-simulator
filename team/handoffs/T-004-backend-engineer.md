# T-004 — backend-engineer — Engine foundations

**Status:** done
**Next owner:** qa-engineer — verify T-004 (start_age > 1, CRN determinism, noise-off equivalence, perf). frontend-engineer can start T-006 against the contract below.

## What I did
- `Stint.start_age: int = 1` (validated `>= 1` in `__post_init__`). `simulate_race` runs tire ages `start_age .. start_age + laps - 1` per stint, calling `predict_delta` with the real age (tire_model.py untouched). Pit loss is now keyed on "first lap of a stint after the first", not `tire_age == 1`, so a stop onto a used set is still charged.
- Monte Carlo vectorized. New `monte_carlo.simulate_totals(strategies, ...) -> Dict[str, np.ndarray]` is the core. Both noise terms in the current model are purely additive, so `noisy total = deterministic total + sum(lap noise) + sum(pit noise)`. The deterministic part is one `simulate_race` call per strategy, which stays the single-run reference. The noise is drawn as matrices and summed with numpy. `run_monte_carlo` and `compare_strategies` both wrap it. Public signatures are unchanged. The docstring warns that this decomposition must be revisited once noise interacts with state (SC, reactive pits).
- Common random numbers: one `(n_sims, max_laps)` lap-noise matrix and one `(n_sims, max_stops)` pit-noise matrix per call, shared by all strategies. Lap j gets the same draw in every strategy, and so does the k-th stop. Draw order is lap noise, then pit noise, from one `default_rng(seed)`. Exact ties go to the first-listed strategy (`np.argmin`), the same rule as the old `min(names, ...)`.
- API: `StintIn.start_age` (optional, default 1, `ge=1`), passed through to `Stint`.

## Files changed
- backend/app/core/race_simulator.py: `Stint.start_age`, lap loop uses it, pit-loss trigger fixed for used sets
- backend/app/core/monte_carlo.py: `simulate_totals` (vectorized, CRN), `run_monte_carlo`/`compare_strategies` reimplemented on top of it, vectorized win counting
- backend/app/api/routes/simulation.py: `StintIn.start_age`
- backend/tests/test_race_simulator.py: 6 start_age tests (default, constant offset on linear curve, cliff crossing, pit loss onto used set, used first stint, start_age=0 rejected)
- backend/tests/test_monte_carlo.py: 8 tests (noise-off == simulate_race for simulate_totals, compare_strategies and run_monte_carlo; noisy path == hand-built per-lap reference loop with the same draws; identical strategies get identical draws; same-length/same-stops difference is exactly deterministic; draws don't change when a non-widening competitor is added; seeded reproducibility)
- backend/tests/test_api_simulation.py: 2 tests (start_age accepted and slows the stint, start_age=0 gives 422)

## Verification
- `pytest` (repo root): 37 passed before, **53 passed** after.
- `validate_replay.py`: **unchanged**. China +26.7s (+0.50%), Britain -150.2s (-2.96%), Austria -40.3s (-0.77%). That's expected, since replays use new sets (start_age=1) until T-002 changes `reconstruct_strategy`.
- Noise-off: vectorized totals match `simulate_race` to atol 1e-9 (tested).
- Perf: `compare_strategies`, 10,000 sims x 3 strategies (M25/H32, S15/M20/H22, H30/M27), real fitted tire model, seed 42: **7.7s before, 0.013s after (~600x)**. Bench script: C:\Users\rushi\AppData\Local\Temp\claude\C--Users-rushi-Desktop-Folders-Code-F1-strategy-simulator\bbb5655a-bfbb-4671-b4b7-bfaa13ddf821\scratchpad\bench.py
- Not verified: end-to-end `/simulate` latency over HTTP (the noise-free trace and model fit are unchanged, and the first request still pays `build_tire_model`). I didn't check the 20-car target either, because a multi-car sim doesn't exist yet.

## Interface / contract changes
**Python:**
- `Stint(compound: str, laps: int, start_age: int = 1)`. `start_age < 1` raises ValueError.
- New `monte_carlo.simulate_totals(strategies: Dict[str, List[Stint]], base_laptime, tire_model, n_simulations=10_000, lap_time_noise_std=0.8, pit_loss_noise_std=5.0, pit_loss_seconds=23.5, seed=None) -> Dict[str, np.ndarray]`.
- `run_monte_carlo`, `compare_strategies` and `simulate_race` signatures are unchanged.

**HTTP `POST /simulate` request (for frontend-engineer, T-006).** Only `StintIn` changed and the change is backward compatible. Omitting `start_age` behaves exactly as before.
```
SimulateRequest {
  base_laptime: float            // seconds, > 0
  strategies: StrategyIn[]       // >= 1, names unique
  n_simulations?: int            // default 10000, 1..50000
  lap_time_noise_std?: float     // seconds, >= 0, default 0.8
  pit_loss_noise_std?: float     // seconds, >= 0, default 5.0
  pit_loss_seconds?: float       // seconds, >= 0, default 23.5
  seed?: int | null
}
StrategyIn { name: string; stints: StintIn[] /* >= 1 */ }
StintIn {
  compound: "SOFT" | "MEDIUM" | "HARD"
  laps: int                      // > 0
  start_age?: int                // NEW. laps of tire age on the stint's first lap; >= 1; default 1 (= new set).
                                 // A set already run N laps -> start_age = N + 1.
}
```
`start_age: 0` (or any value < 1) returns 422. The response shape is **unchanged**: `{ results: [{ name, mean, std, p10, p50, p90, win_probability, lap_times: float[], pit_laps: int[] }] }`. A stop onto a used set still appears in `pit_laps`.

## Requests for other roles
- data-scientist (T-002): once `reconstruct_strategy` detects used-set stints, set `Stint(compound, laps, start_age=<tire_age of first lap>)` so replays use this. Note that `predict_delta` is anchored at age 2 and clamped monotone. Large start ages run into the max_reliable_age/cliff extrapolation (flat placeholder penalty), so it's worth checking how often used-set ages exceed that.
- frontend-engineer (T-006): build against the `StintIn.start_age` contract above. Expect win_probability values to be much more decisive than before (see the first risk below). Don't edit anything backend-side.

## Open questions / risks
- **Visible behavior change in win_probability.** With CRN, strategies with the same race length and the same number of stops share every draw, so their ranking is deterministic. Example: 1-stop M25/H32 vs 1-stop H30/M27 went from 43%/46% to 0%/98%. Only pit-noise draws for stops one strategy makes and the other doesn't can flip a ranking. This is the correct consequence of iid additive lap noise (ROADMAP point 3: that noise carries no strategy information), and win probability will only become meaningful once noise interacts with race state (SC, traffic, Phase 1/2). The orchestrator may want the UI copy to say this.
- Seeded noisy outputs differ numerically from before T-004 because the draw order changed. Noise-off outputs are identical.
- `simulate_totals` relies on noise being additive and state-independent. Phase 1/2 (traffic, SC) will break that assumption and needs a genuinely vectorized lap loop over `(n_sims,)` state arrays. This is documented in the docstring.
- `cache/fastf1_http_cache.sqlite` shows as modified after running validate_replay.py. That comes from FastF1, not from my code changes.
