# T-017 — backend-engineer — GET /tracks, POST /optimize, track_id on /simulate

**Status:** done
**Next owner:** qa-engineer — verify; then frontend-engineer (T-018) builds against the models below. data-scientist: see requests.

## What I did
- `track_params.py`: single loader (`load_tracks`, `get_track`, `set_track_params_path`, env `TRACK_PARAMS_PATH`), default `data/tracks/track_params.json`, cached per path; missing/invalid file -> `TrackParamsUnavailable` -> HTTP 503 with a message pointing at `build_track_params.py`.
- `optimizer.py`: enumerate plans (1..max_stops stops, >=2 distinct dry compounds, stint >= 5 laps, stint <= `max_stint_laps[compound]`, new tyres only) on a length grid (step 1/2/3 for 2/3/4 stints; last stint takes the remainder), screen all with an O(1)-per-stint closed form that equals `simulate_race` (asserted in a test), hill-climb (+-1/+-2 lap moves) the best plan of each compound sequence, then the best 20 go through `monte_carlo.simulate_totals` (CRN, pit noise = track `pit_loss_noise_std_s` else 5.0, lap noise 0.8). Ranked by mean; paired deltas vs rank 1.
- Routes in `api/routes/tracks.py` (+ `api/deps.py` for shared `resolve_track`), registered in `main.py`.
- `/simulate`: optional `track_id`. `base_laptime` and `pit_loss_seconds` are now Optional in the schema; resolution is explicit value > track value > global default (`pit_loss_noise_std` takes the track value only if the client did not send it). Without `track_id` behaviour is identical (existing tests pass; `base_laptime` missing and no track is still a 422).
- Did not touch `race_simulator.py`, `monte_carlo.py`, or `tire_model.py`.

## Files changed
- backend/app/core/track_params.py (new), backend/app/core/optimizer.py (new)
- backend/app/api/deps.py (new), backend/app/api/routes/tracks.py (new)
- backend/app/api/routes/simulation.py — optional `track_id`, Optional `base_laptime`/`pit_loss_seconds`
- backend/app/main.py — include tracks router
- backend/tests/test_optimizer.py, backend/tests/test_api_tracks.py, backend/tests/fixtures/track_params_fixture.json (new)

## Verification
- `pytest` (repo root): 276 passed (16 new).
- Timing (real tyre model, 56 laps, n_sims=2000): max_stops=1 0.004 s, 2 0.008 s, 3 0.011 s end to end in-process (33,060 grid plans screened for 3 stops; 20 simulated). Far under the 3 s target; test asserts < 3 s both for `optimize()` and the HTTP call.
- Covered: two-compound rule, min stint, caps, pit_laps = cumulative stint ends, ranking sorted by mean and consistent with paired deltas, rank-1 deltas exactly 0, seed determinism, local optimality of the winner, 404 unknown track, 503 missing file, 422 on all bounds, 422 when caps cannot cover the distance, /simulate with track defaults equals the explicit-values request, legacy /simulate unchanged.
- NOT verified: against the real `data/tracks/track_params.json` (did not exist yet; tests use the fixture). `validate_replay.py` not run (simulation loop untouched).

## Interface / contract changes
Exact models (pydantic, `backend/app/api/routes/tracks.py`):

```python
class TrackOut(BaseModel):            # extra="allow"
    track_id: str; name: str; circuit: Optional[str]
    total_laps: int; pit_loss_s: float; pit_loss_noise_std_s: Optional[float]
    base_laptime_s: float
    compounds: Optional[Dict[str, Optional[str]]]
    max_stint_laps: Dict[str, int]
    source_season: Optional[int]; source_races: List[str]
    on_2026_calendar: Optional[bool]; already_raced_2026: Optional[bool]; notes: Optional[str]

class OptimizeRequest(BaseModel):
    track_id: str
    max_stops: int = 3            # 1..3
    n_sims: int = 2000            # 1..20000
    top_n: int = 5                # 1..10
    total_laps: Optional[int]     # 1..100, null = track value
    pit_loss_s: Optional[float]   # 0..120
    base_laptime_s: Optional[float]  # >0, <=300
    seed: Optional[int] = 0

class StintOut(BaseModel): compound: str; laps: int; start_age: int   # start_age always 1
class StrategyOut(BaseModel):
    rank: int; n_stops: int; stints: List[StintOut]
    pit_laps: List[int]           # last lap of each stint except the final (pit at end of that lap)
    mean_s: float; p10_s: float; p90_s: float
    delta_to_best_mean_s: float; delta_to_best_p10_s: float; delta_to_best_p90_s: float
class ParamsUsed(BaseModel): total_laps: int; pit_loss_s: float; base_laptime_s: float; n_sims: int
class OptimizeResponse(BaseModel):
    track: TrackOut; params_used: ParamsUsed; strategies: List[StrategyOut]
    n_candidates_screened: int; n_candidates_simulated: int; assumptions: List[str]
```
- `GET /tracks` -> `List[TrackOut]`; 503 if file missing.
- `POST /optimize`: 404 unknown `track_id`; 422 bounds or infeasible (caps too small for distance); 503 file missing.
- `POST /simulate` additions: `track_id?: string`; `base_laptime?: number` (required only if no `track_id`); `pit_loss_seconds?: number` (default 23.5, or the track's). Response unchanged. Note: with `track_id` the strategies' stint laps are NOT forced to equal the track's `total_laps`.

## Requests for other roles
- data-scientist: contract says `base_laptime_s` is "fuel-corrected to lap 1 start" (i.e. heavy car), but `race_simulator` treats base_laptime as a fully light car (fuel effect anchored to 0 at the last lap, fuel coef negative per lap). Please confirm which one the file holds; if it is heavy-car pace, either subtract `fuel_coef*(total_laps-1)` in the builder or tell me to convert at load time. Absolute plan times are off by roughly that amount otherwise (ranking is unaffected).
- data-scientist: compound offsets and cliff penalties are still placeholders, so the optimizer currently favours SOFT-heavy plans (on the fixture 56-lap case it picks SOFT 25 + MEDIUM 31, with SOFT capped at its 25-lap max_stint). Fitting those is the biggest quality lever for T-019.
- frontend-engineer: show `assumptions` verbatim; deltas under ~5 s are model noise (the response does not flag this).

## Open questions / risks
- Rank 1 and 2 can be the same compounds in swapped order with near-identical times (fuel-only difference); the UI may want to treat them as separate plans (they are).
- Plans with two stints on the same compound (e.g. M/S/S) are allowed by the rule and may appear.
- Used-set starts (`start_age > 1`) and the actual starting-compound constraint are not modelled in the optimizer.

## Follow-ups
- Reordered duplicates removed: finalists are deduped by sorted (compound, laps) multiset, keeping the best-ordered plan; `n_candidates_simulated` is the deduped count. `n_candidates_screened` is still the grid size.
- Caps floored: cap = max(observed `max_stint_laps`, 15) (still clipped to total laps); stated in `assumptions`. The `track` object in the response still shows the observed values.
- Resolved: `base_laptime_s` follows the simulator convention (HARD, age 2, final-lap fuel). The earlier request to the data-scientist about fuel correction no longer applies; contract wording fixed.
- Tests added: `test_no_reordered_duplicates`, `test_cap_floor`.


## T-021 (min stint + effective caps)
- optimizer.py: MIN_STINT_LAPS 5 -> 8, plus `min_stint_laps(total_laps)` = max(8, round(10% of laps)), clamped to laps//2. Applies to every stint (first/last too) in grid, hill-climb and feasibility check. Rationale in code comment; new assumption line added to `ASSUMPTIONS`. Two-compound rule unchanged. Same-compound repeats not penalised.
- API (tracks.py): `params_used` gains optional `effective_max_stint_laps: {SOFT|MEDIUM|HARD: int}` (post 15-lap floor and clip to race laps; only compounds with caps and a fitted model) and `min_stint_laps: int`. Existing fields unchanged. Contract doc updated. FRONTEND: add both as optional to `ParamsUsed` in lib/api.ts and show the effective cap next to the observed one (and note min stint) in the UI.
- Tests: updated test_api_tracks params_used asserts (subset compare); added 4 tests in test_optimizer.py (property-based, no pinned fitted numbers; real-model runtime < 3 s and determinism).
- Tests that the data-scientist refit could legitimately change: none pinned to fitted values by me; test_t019_qa sanity checks should be rerun after refit.
