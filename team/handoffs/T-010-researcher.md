# T-010 — researcher — US GP 2026 brief

**Status:** done
**Next owner:** data-scientist — use the brief for T-011 feature choices and sanity checks; orchestrator — resolve the FastF1 access risk below before Saturday.

## What I did
- Wrote `team/research/T-010-us-gp-2026.md`: weekend schedule, COTA track character (2022-25 stops, pit-lane times, SC/VSC signatures, 2012-25 grid-vs-finish), 2025 to 2026 transfer (team points shares), predictive-power numbers for 2026 and 2022-26, calendar check, dated standings, T-011 recommendations.
- Data pulled from Jolpica (results 2022-26, COTA 2012-25, COTA pit stops/laps 2022-25) into my scratchpad only; FastF1 cache was only read (schedule).

## Files changed
- `team/research/T-010-us-gp-2026.md` — research note
- `team/handoffs/T-010-researcher.md` — this handoff

## Verification (key numbers)
- 2026 (17 races): mean Spearman(grid, finish) 0.647; pole won 12/17 (70.6%); winner from top 3 in 16/17 (Antonelli won from P19 at Monza); DNF+DNS 20.9% of starts vs 12.5% in 2025.
- Pooled 2022-26: grid-1 win rate 58% (n=109), podium 81%; COTA winners from P1/P2 in 11 of 13 races, pole won 6/13; COTA mean grid-vs-finish rho 0.52 (2021-25: 0.49).
- Baselines vs finishing order in 2026: grid 0.65, pre-race championship points 0.63, last-3 avg finish 0.59, last-race finish 0.45.
- Team points share 2025 to 2026: McLaren 32.0% to 16.6%, Mercedes 17.5% to 29.9%, Ferrari 14.9% to 22.3%.
- Calendar: round 16 "Bahrain GP / Kuala Lumpur" is real: Bahrain moved to Sepang, 2-4 Oct 2026 (war-related; Saudi cancelled).
- Standings (Crash.net, 2026-10-11): ANT 340, RUS 238, HAM 229, VER 221, LEC 215, NOR 201. My race-points-only Jolpica totals reproduce the order.
- NOT verified: Pirelli's official 2026 US GP nominations (C2/C3/C4 from one Italian source); causes of 2023/2024 slow laps at COTA; COTA degradation in s/lap; Austin weather; data-availability lags after sessions; any published practice-long-run result.

## Interface / contract changes
- none

## Requests for other roles
- orchestrator: **FastF1 live-timing loads failed from my sandbox** (HTTP 401 on `livetiming.formula1.com`, all `Session.load` endpoints "Failed to load", fresh scratch cache, 2025 US GP). Jolpica worked. Test a brand-new-session FastF1 pull from your normal shell before Friday 2026-10-23, and keep Jolpica as a fallback for the post-penalty grid. Also, the 2025 US GP (R19) was not yet in `cache/2025` when I checked; the COTA 2025 lap-level data would let T-011 fit a per-track degradation prior.
- data-scientist (T-011): (1) use `grid` with penalties, treat DNS separately from DNF, 22 starters in 2026; (2) model DNF risk explicitly (team rates 6-50% in 2026); (3) tag Sepang as its own circuit and Barcelona vs Madrid as different circuits; (4) shrink the 2026 pole-win rate (71%, n=17) toward the pooled 58% and COTA 46%; (5) use the section 7 sanity ranges for the Saturday table; (6) down-weight 2025 team pace to zero, keep 2025 only for track-level and driver priors.
- data-scientist (T-009): include a flag `event_format` (US GP 2026 is conventional; 2025 was sprint); keep FP2/FP3 laps if cheap to pull, since FP long-run value is untested.

## Open questions / risks
- Rain at Austin is unknown (no forecast found); a wet race would break the dry-based features.
- Antonelli may clinch the title in Austin; no evidence of impact on racing, but possible Mercedes team-order effect.
- Season-end rounds (Qatar/Abu Dhabi) were reported at risk; irrelevant to the US GP.
- Sample sizes are small throughout (17 races, 13 COTA races); report intervals.
