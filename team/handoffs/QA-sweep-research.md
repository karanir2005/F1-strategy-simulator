# QA-sweep — researcher — pre-Austin (R18) fact check

**Status:** done
**Next owner:** orchestrator: write `data/jolpica/grid_override_2026_R18.csv` after qualifying (see 3). Data-scientist/backend: Austin HARD compound fix (see 2), not urgent for the winner prediction.

## 1. Sprint? No. Our data is correct.
- 2026 Austin uses the normal format: FP1 and FP2 Fri 23 Oct, FP3 and Quali Sat 24 Oct 16:00 CDT (21:00 UTC), race Sun 25 Oct 15:00 CDT, 56 laps. [F1 tyre preview, 21 Oct 2026](https://www.formula1.com/en/latest/article/what-tyres-will-the-teams-and-drivers-have-for-the-2026-united-states-grand-prix.44LKddYSbUJK1LxMAa6VLI) ("The usual Grand Prix format is in place"), [RacingNews365 timetable](https://racingnews365.com/2026-f1-united-states-grand-prix-austin-time-schedule).
- The 2026 sprint venues were Shanghai, Miami, Montreal, Silverstone, Zandvoort and Singapore ([FIA](https://www.fia.com/news/fia-and-formula-1-announce-2026-sprint-calendar)).
- `data/jolpica/derived/schedule.csv` line 111: `2026,18,...,americas,2026-10-25,False,2026-10-24,21:00:00Z`. Sprint=False and the quali time match. 2023-2025 Austin are correctly True. Nothing to change.

## 2. Tyres: our HARD is WRONG for 2026 (C1 should be C2)
- 2026 allocation: **HARD C2 / MEDIUM C3 / SOFT C4** (same F1 article). 2025 was C1/C3/C4, a non-consecutive pick at a sprint weekend.
- `data/tracks/track_params.json` `austin` has `HARD: "C1"` and all its wear/stint data from `Austin_2025` (2025 cars, 2025 tyre construction, sprint weekend). Expect a less durable HARD that is closer to MEDIUM pace than the 2025 numbers suggest. `max_stint_laps.HARD=28` is a 2025 artefact. This affects the `/strategy` page only, not the winner predictor. Fix the label and add a note. The 2025 HARD wear numbers are a prior, not 2026 truth.

## 3. Grid penalties: three announced (FIA, 23 Oct 2026)
[F1.com](https://www.formula1.com/en/latest/article/lindblad-colapinto-and-bortoleto-to-take-grid-penalties-for-united-states-grand-prix.4qjn84W4zOmwNXeELFWn0k), also [PlanetF1](https://www.planetf1.com/news/us-grand-prix-grid-penalties-colapinto-bortoleto-lindblad):
- **BOR** (Audi): 4th ES + 4th CE, -20 places, so effectively the back of the grid.
- **COL** (Alpine): 4th CE, -10 places from his quali result.
- **LIN** (Racing Bulls): 6th ICE, -5 places.
- **Impact:** grid mode uses the qualifying classification and does **not** apply penalties (`prediction_service.py` L114) unless `data/jolpica/grid_override_2026_R18.csv` (`code,grid`; 0 = pit lane) exists. These are midfield drivers, so win/podium odds barely move. Expected grid and Q-mode numbers for everyone classified behind them will be off by 1-3 places. More penalties can still come on Saturday or Sunday (12 of 22 drivers are already over at least one allocation, per PlanetF1). Build the override from the FIA's official starting grid, not by hand.
- **Lineup:** FP1/FP2 entry ([The Race](https://www.the-race.com/formula-1/f1-2026-united-states-grand-prix-practice-results/)) matches `results.csv` R17 exactly (22 drivers, 11 teams incl. Cadillac PER/BOT, Audi HUL/BOR). Colton Herta drove Pérez's Cadillac in FP1 only. Pérez races. I could not reach the API (`127.0.0.1:8001` refused / not fetchable), so I checked results.csv instead.

## 4. Pole-to-win and safety cars at COTA
- **Pole won 6 of 13** COTA races 2012-2025 (2013, 16, 17, 19, 21, 25), about 46%. Since 2018 (the T-037 window, n=7) it is 3 of 7 (43%). In our Jolpica data 2022-25 it is 1 of 4 (only VER 2025; 2023 pole LEC was DSQ, 2024 pole NOR finished P4). Winners started P2, P6 and P4 in 2022-24. Sources: [LightsOut grid-slot stats](https://lightsoutblog.com/f1-grid-slot-stats-united-states-grand-prix-cota) (5 pole wins 2014-25), FIA 2022 preview (pole won 5 of the first 9). The 2013 pole win is from memory plus the FIA count, so treat it as lightly verified.
- 2026 so far, from results.csv: pole won **12 of 17 (71%)**. A multiplier of **0.80 is directionally right**: Austin really is below average for pole conversion. It rests on few races, and T-037 shows it does not change the pick.
- **Safety cars (races only, 2012-25, n=13; [LightsOut SC list](https://lightsoutblog.com/safety-car-deployments-united-states-grand-prix)):** full SC in 2013, 2015, 2022 and 2024 (2014 is ambiguous on the page, so **unverified**), which is 4-5 of 13 (31-38%). VSC in 2015, 2016, 2018, 2021 and 2025, which is 5 of 13 (38%). No neutralisation at all in 2012, 2017, 2019 and 2023. Our `safety_car_params.json` is pooled over 40 races (P(any SC) 0.525, P(any VSC) 0.50), so it is about 1.4-1.7x too SC-heavy for COTA. VSC is about right. Our `Austin_2025` entry (VSC lap 7, 2 laps, no SC) matches the source. This is a strategy-sim bias only, not prediction-breaking. All of this history is from pre-2026 regulations.

## Nothing found that breaks tomorrow's winner prediction
The one action item is the post-penalty grid override for R18. Without it, the qualifying grid is wrong for BOR/COL/LIN and everyone behind them.
