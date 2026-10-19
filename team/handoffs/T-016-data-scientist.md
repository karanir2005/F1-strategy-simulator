# T-016 — data-scientist — Per-track parameters (track_params.json)

**Status:** done
**Next owner:** qa-engineer — verify; backend-engineer (T-017) can read `data/tracks/track_params.json` now.

## What I did
- `build_track_params.py` (repo root, offline, ~2 s) writes `data/tracks/track_params.json`: 23 objects, one per 2026 calendar circuit (Jolpica schedule order), schema exactly as contract section 1.
- 17 circuits use 2026 data (Australia..Singapore, incl. Madrid/Madring and Malaysia/Sepang); 6 use 2025 (Austin, Mexico, Brazil, Las Vegas, Qatar, Abu Dhabi), flagged in `notes`. `already_raced_2026` is true for 17.
- Methods are in the script docstring. Key points:
  - total_laps: Jolpica winner laps (matched race_info everywhere).
  - pit_loss_s: median pit_lane_s of tyre-change stops not under SC/VSC/red, 10-120 s. If the 2026 race has <15 clean stops (China, Japan, Italy, Azerbaijan, Singapore) the 2025 stops are pooled (noted). noise_std = std of those stops <= 60 s.
  - base_laptime_s: clean-air (gap >= 2 s), `tire_model.clean_laps` laps of the 5 fastest drivers, corrected for tyre age, compound offset and fuel with the fitted tire model, median.
  - max_stint_laps: longest observed stint per dry compound (split at real pit stops, compound changes, tyre-age resets). Gaps filled from the other season (noted). Monaco uses Monaco_2025 (2026 red flag).
- Excluded sources: Belgium_2025, Australia_2025, Britain_2025 (wet/phantom), anything usable=false (Miami_2025, Australia_2026 staging; Australia_2026 uses the FastF1 processed file).

## Files changed
- build_track_params.py — new
- data/tracks/track_params.json — new output
- backend/tests/test_track_params.py — new (6 tests: schema, all 23 calendar circuits, ranges, 2025 flags, Austin vs contract)

## Verification
- `python build_track_params.py` OK; `pytest backend/tests/test_track_params.py` 6 passed.
- Austin: 56 laps, pit 24.0 s (std 0.8), base 98.11 s, compounds C1/C3/C4 (from T-010 research, since OpenF1 files lack C-numbers).
- Not verified: no replay of base_laptime_s through simulate_race against real race times; no full-suite run.

| track | laps | pit loss | base lap | season |
|---|---|---|---|---|
| australia | 58 | 19.1 | 81.36 | 2026 |
| china | 56 | 22.9 | 94.38 | 2026 |
| japan | 53 | 23.5 | 92.34 | 2026 |
| miami | 57 | 23.2 | 91.07 | 2026 |
| canada | 68 | 24.9 | 74.20 | 2026 |
| monaco | 78 | 25.4 | 74.14 | 2026 |
| barcelona | 66 | 22.6 | 80.20 | 2026 |
| austria | 71 | 21.6 | 69.76 | 2026 |
| britain | 52 | 29.5 | 92.29 | 2026 |
| belgium | 44 | 24.7 | 108.91 | 2026 |
| hungary | 70 | 22.0 | 82.61 | 2026 |
| netherlands | 72 | 18.0 | 74.13 | 2026 |
| italy | 53 | 25.0 | 83.38 | 2026 |
| madrid | 57 | 31.4 | 96.12 | 2026 |
| azerbaijan | 51 | 20.6 | 106.12 | 2026 |
| malaysia | 55 | 29.1 | 100.44 | 2026 |
| singapore | 62 | 24.4 | 95.70 | 2026 |
| austin | 56 | 24.0 | 98.11 | 2025 |
| mexico | 71 | 22.7 | 80.50 | 2025 |
| brazil | 71 | 23.5 | 72.42 | 2025 |
| las_vegas | 50 | 21.4 | 93.84 | 2025 |
| qatar | 57 | 28.5 | 83.57 | 2025 |
| abudhabi | 58 | 21.8 | 86.73 | 2025 |

## Interface / contract changes
- **base_laptime_s convention differs from the contract wording.** The contract says "fuel-corrected to lap 1 start"; `simulate_race` instead expects a HARD, age-2, final-lap-fuel (light car) reference (`fuel_effect` is anchored to 0 at the last lap). I followed the simulator so the value can be passed straight in as `base_laptime`. Flag for the orchestrator/backend: no conversion needed.
- compounds is null for 14 circuits (C-numbers not in OpenF1 data); Austin is filled from research. Per-compound values are never null where the dict exists except none currently.
- max_stint_laps omits a compound never run (Las Vegas SOFT); optimizer must skip it.

## Requests for other roles
- backend-engineer: treat missing max_stint_laps keys as "compound unavailable"; handle `compounds: null`.

## Open questions / risks
- Several caps are small/odd because they are observed maxima, not tyre limits (China SOFT 9, Abu Dhabi SOFT 8, Britain SOFT 12, Qatar 25 = the race tyre-life rule). They bound the optimizer; consider a floor or global minimum cap.
- Pit-loss std for Australia (7.1), China (5.3), Britain (5.0) reflects few stops / slow outliers within 60 s.
- 2025-sourced base lap times are 2025-car pace and could be off for 2026 cars; base lap depends on placeholder pace offsets and the 10-race tire model fuel coefficient.
- Malaysia/Madrid/Singapore stints come from OpenF1 staging with known tyre-record gaps (handled, still lower confidence).
