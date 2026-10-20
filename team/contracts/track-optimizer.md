# Contract: track selection + optimal stint plan (T-016 to T-019)

User request (2026-10-19): in Race Settings, pick a track instead of typing numbers; the sim uses that track's real lap count, pit loss and pace; and the model returns the optimal stint plan for each track.

## 1. Track parameters file (T-016, data-scientist)

`data/tracks/track_params.json`, built by `build_track_params.py` (repo root, offline, from `data/processed/` + `data/staging/` + cached Jolpica). A list of objects:

```json
{
  "track_id": "austin",            // stable slug
  "name": "United States GP (Austin)",
  "circuit": "Circuit of the Americas",
  "total_laps": 56,
  "pit_loss_s": 21.3,              // median green-flag pit-lane duration, red-flag/>120 s stops excluded
  "pit_loss_noise_std_s": 3.1,
  "base_laptime_s": 98.4,          // representative clean-air race pace for a front-running car, simulator convention: HARD compound, tyre age 2, final-lap (light) fuel, i.e. what race_simulator takes as base_laptime
  "compounds": {"HARD": "C1", "MEDIUM": "C3", "SOFT": "C4"},   // null if unknown
  "max_stint_laps": {"SOFT": 25, "MEDIUM": 35, "HARD": 45},    // longest observed stints (upper bound for optimizer caps)
  "typical_stint_laps": {"SOFT": 22, "MEDIUM": 30, "HARD": null},  // OPTIONAL (T-026): p90 of observed stint lengths per compound, null if < 5 stints. Optimizer cap = min(max, typical) when present, then floor 15 / clip to race laps. Absent in old payloads.
  "source_season": 2026,           // 2026 if raced in 2026, else 2025 (track-dependent params only)
  "source_races": ["Austin_2025"],
  "on_2026_calendar": true,
  "already_raced_2026": false,
  "notes": "..."
}
```

Cover every 2026 calendar circuit (raced or upcoming: Austin, Mexico, Brazil, Las Vegas, Qatar, Abu Dhabi use 2025 data). Skip unusable races (`usable=false` in race_info). Pit loss and lap count are track properties, so 2025 is fine for them; base lap time from 2025 must be flagged in `notes`.

## 2. API (T-017, backend-engineer)

`GET /tracks` returns the list above (read from the JSON at startup; 503 with a clear message if missing).

`POST /optimize`:
```json
request:  {"track_id": "austin", "max_stops": 3, "n_sims": 2000, "top_n": 5,
           "total_laps": null, "pit_loss_s": null, "base_laptime_s": null, "seed": 0}
response: {"track": {...track params...},
           "params_used": {"total_laps":56, "pit_loss_s":21.3, "base_laptime_s":98.4, "n_sims":2000},
           "strategies": [{"rank":1, "n_stops":1,
                           "stints":[{"compound":"MEDIUM","laps":24,"start_age":1},{"compound":"HARD","laps":32,"start_age":1}],
                           "pit_laps":[24], "mean_s":..., "p10_s":..., "p90_s":...,
                           "delta_to_best_mean_s":0.0, "delta_to_best_p10_s":..., "delta_to_best_p90_s":...}],
           "n_candidates_screened": 1234, "n_candidates_simulated": 20,
           "assumptions": ["single car in clean air", "no safety cars", "global tyre-wear curves", ...]}
```
`params_used` also carries two OPTIONAL fields (T-021): `effective_max_stint_laps: {compound: int}` (stint cap actually used per compound = observed max floored to 15 and clipped to race laps; may differ from the track's observed `max_stint_laps`) and `min_stint_laps: int` (minimum length of every stint, = max(8, 10% of race laps), clamped to laps//2).
Nulls mean "use the track's value". `max_stops` 1–3, `n_sims` ≤ 20000, `top_n` 1–10.

Optimizer: enumerate compound sequences with ≥2 distinct dry compounds (the rule), 1..max_stops stops, stint lengths within `max_stint_laps` (no extrapolation past observed data); deterministic screen with `race_simulator`, then Monte Carlo with common random numbers on the best ~20; rank by mean, report paired deltas vs rank 1. Must answer in < 3 s for max_stops=3.

`POST /simulate` also accepts an optional `track_id` (fills total_laps / pit loss / base lap time when not given). Existing behaviour unchanged when omitted.

## 3. Frontend (T-018)
Race Settings: track dropdown (from `GET /tracks`), showing laps, pit loss, base lap time, compounds and source season, with an "override" toggle for the numbers. A "Find optimal strategy" button calls `/optimize` and shows the top plans (stint bar per plan, pit laps, delta to best with p10–p90) and the assumptions list. Manual strategy comparison stays.
