# T-001 — researcher — FastF1 schema + 2026 rules brief

**Status:** done
**Next owner:** data-scientist — T-002. Implement the stint/tyre-change detection, gap and flag recipes, and the column list from `team/research/T-001-fastf1-schema-2026-rules.md` §2 and §4.

## What I did
- Loaded all 10 cached 2026 races read-only (`offline_mode(True)`, scripts in the scratchpad). Checked column presence and null rates, `Stint`/`FreshTyre`/`TyreLife` consistency, `Position` against `Time` order, final-lap gaps against `session.results`, TrackStatus codes, race control messages, weather, and the `car_data` DRS channel.
- Pulled 2026 rules from the FIA Sporting Regulations Issue 07 (25 Jun 2026) PDF: tyres B6, Overtake B7, VSC/SC/suspension B5, pit speed B1.6. Added per-race C-compound mappings from team/Pirelli previews.

## Files changed
- team/research/T-001-fastf1-schema-2026-rules.md: research note (new)
- team/handoffs/T-001-researcher.md: this handoff (new)

## Verification
- `Stint` increment ⇔ `PitInTime` on the previous lap: 387/387, 0 exceptions both ways.
- Same-tyre `Stint` increments (pit-lane pass, drive-through, garage): 44. Real tyre changes: 343, of which 57 were onto used sets (start age 2–9, mostly 4–5).
- Same-lap gap from `Time` vs `results.Time` on the final lap: median abs error 0.00–0.09 s per race. Outliers of 5/10/20 s are post-race penalties.
- `Position` == rank of `Time` within the lap: 96.8–100%.
- `car_data.DRS`: 0 in all 33,508 samples (Britain, LEC).
- **Not verified:**
  - the Detection Gap value per circuit (secondary sources say 1.0 s)
  - Monaco's 60 km/h pit limit
  - 2026 race fuel load
  - the Miami compound mapping (secondary source only)
  - Spain, which is not in the cache

## Interface / contract changes
- None made. Proposed laps CSV schema and sidecar CSVs: see note §4. Existing columns are kept. `lap_time_seconds` becomes nullable, so the null filter moves from extraction to `clean_laps`.
- `stint_start_age` maps directly to T-004's `Stint.start_age` (= `TyreLife` on the stint's first lap).

## Requests for other roles
- data-scientist (T-002):
  - Split `Stint` increments into real tyre changes and same-tyre pit passes (§2.1). Otherwise Monaco replays get ~35 phantom stops.
  - Add `compound_c` from the §3.2 table.
  - Fix the Spain/Barcelona ambiguity in `pull_races.py`. `"Spain"` resolves to the Madrid race, and the re-pull needs network access.
  - Drop DNS drivers and `FastF1Generated` rows.
- data-scientist (T-003): score the finishing order against `results.Position`, because `results.Time` includes post-race penalties.
- backend-engineer (Phase 1/2, later): the Overtake eligibility gap is to **any** car ahead on the road, lapped cars included. Overtake is disabled on lap 1 and under SC until the restart. A red flag is a free tyre change. Under VSC the pit lane may be entered only to change tyres.

## Open questions / risks
- Orchestrator decision: should T-002 re-pull Barcelona, Madrid, or both? The Spain row in `pull_races.py` currently points to Madrid and was never cached.
- Pooling S/M/H labels across races mixes C3 and C5 under "SOFT". This is a probable cause of the inverted SOFT offset, which is a Phase 3 concern but cheap to record now.
- Monaco 2026 (SC through the pit lane twice, plus a red flag with free tyre changes) is an outlier for pit-loss calibration.
