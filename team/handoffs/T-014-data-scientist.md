# T-014 — data-scientist — Per-race 2026 table + per-track grid-stickiness term

**Status:** done. The per-race table is delivered. The stickiness term is built, leakage-tested and **OFF by default** (`stick_gamma=0.0`): it did not improve the 2025 tuning season, so by the pre-committed rule it is not kept.
**Next owner:** qa-engineer (review the new code and tests), then the orchestrator.

## What I did
1. **Per-race walk-forward table** (grid mode, rounds 1-17, each race predicted from earlier rounds plus its own grid, default config, 10k sims): `data/jolpica/predictions/walk_forward_2026_by_race.csv`. Regenerate with `python predict_race.py --by-race --offline --sims 10000`.
   - Columns: predicted vs actual winner and podium, P(actual winner), winner NLL, win/podium/top10 log-loss, Spearman(grid, result), plus the stickiness values used.
   - Rounds 2-17 reproduce T-011 exactly: win LL .0790, podium LL .2075, top10 LL .4839, winner NLL 1.103.
   - Round 1 is included, with no 2026 history and only the 2025 coefficient prior.
2. **Stickiness term** in `predictor.py`:
   - **Per-race measure:** Spearman(grid, finishing position) among classified cars, grid 0 treated as last.
   - **Circuit estimate:** a weighted mean over earlier races at the circuit, shrunk toward the global weighted mean: `s_c = (sum w*rho_c + k*s_global) / (sum w_c + k)`. Races of the target season weigh 1 and earlier seasons weigh `stick_w_old` = 0.5 (era-mixed; the output columns `stick_n_current_season` / `stick_n_earlier_seasons` show the mix).
   - **Where it enters:** it scales the grid term of the utility, `u = b_race*theta + b_grid*m*(-log grid)` with `m = exp(gamma*(s_c - s_global))`. The same `m` is applied to each training race when `b_grid` is fitted.
   - **Walk-forward:** only races with (season, round) strictly before the target are used. A brand-new circuit (Madrid on its first appearance) or no history gives m = 1 exactly.
   - **New config fields:** `stick_gamma` (0 = off), `stick_k` (4), `stick_w_old` (0.5).
   - **Circuit history:** Jolpica 2018-2021 results fetched (4 seasons, about 18 requests) into `data/jolpica/derived/results_2018_2021.csv`. 2022-26 was already cached. The 2018-21 data is used only for the stickiness table, passed as `prepare_history(..., track_results=, schedule=)`. `predict_race.py::load` loads it automatically.

## Tuning (2025 only, then 2026 as a report)
2025 walk-forward, rounds 2-24, 5k sims. Each cell is grid | pre-quali:

| gamma, k | win LL | podium LL | winner NLL |
|---|---|---|---|
| 0 (off) | .0881 \| .1336 | .1917 \| .2535 | 1.059 \| 1.755 |
| 3, 2 | .0885 \| .1339 | .1906 \| .2543 | 1.069 \| 1.763 |
| 6, 4 | .0890 \| .1341 | .1907 \| .2548 | 1.076 \| 1.767 |
| 10, 4 | .0900 \| .1347 | .1903 \| .2558 | 1.094 \| 1.779 |
| 15, 4 | .0918 \| .1358 | .1905 \| .2572 | 1.122 \| 1.799 |

- It trades slightly better podium LL in grid mode against worse win LL, winner NLL and all pre-quali metrics. The win+podium LL sum is flat in grid mode and worse in pre-quali.
- **Decision on 2025: no gain, so off.**

2026 walk-forward, rounds 2-17, shown for information only and not used to choose anything. Each cell is grid | pre-quali:

| gamma, k | win LL | podium LL | winner NLL |
|---|---|---|---|
| 0 (off) | .0795 \| .1126 | .2084 \| .2435 | 1.112 \| 1.629 |
| 3, 2 | .0786 \| .1128 | .2080 \| .2437 | 1.092 \| 1.633 |
| 6, 2 | .0780 \| .1128 | .2072 \| .2440 | 1.075 \| 1.634 |
| 10, 4 | .0781 \| .1129 | .2066 \| .2440 | 1.074 \| 1.635 |

- Grid mode improves by about .001-.002 win LL and .002 podium LL. Pre-quali is slightly worse.
- These gains are small, within what 16 races can resolve, and the 2025 season does not corroborate them. I did not re-pick gamma on 2026.
- **Why the term has little to give:** a single race's Spearman is mostly noise (safety cars, rain, strategy). Circuit means from 3-11 races differ only modestly, and the shrinkage cuts that further.
- 2026 does not always agree with history: Monaco R6 had grid-result Spearman 0.65 vs a historical circuit mean of 0.87. The new cars and regulations may have changed how much overtaking is possible.
- **Decision: stickiness stays off by default.** Enable it with `PredictorConfig(stick_gamma=...)` or `predict_race.py --stick-gamma 3`. The grid-mode gain on 2026 is a lead worth revisiting once more 2026 races exist.

## Biggest misses (grid mode, P of the actual winner)
- **R13 Italian:** ANT won from P19. P = 0.4%, NLL 5.56, the worst by far. A win from the back of the grid is not foreseeable from the grid.
- **R9 British:** LEC won from P2. P = 14.7%, NLL 1.92, against a pole sitter given 63%.
- **R7 Barcelona:** HAM won from P2. P = 21.9%, NLL 1.52.
- **R14 Spanish:** ANT won from P2. P = 23.5%, NLL 1.45.
- **R5 Canadian:** ANT won from P2. P = 31%, NLL 1.17.
- The model picked the actual winner in 12 of 17 races. Every miss except Italian is a P2 starter beating a pole sitter given 41-63%, which is roughly the base rate, so there is no systematic bias.
- Mean Spearman(grid, result) is 0.78. It was lowest at R9 British (0.60), R16 Malaysia (0.61), R6 Monaco (0.65) and R15 Azerbaijan (0.67), and highest at R14 Spanish (0.975).

## COTA (americas) and the US GP pre-quali top 5
- **COTA stickiness:** 7 earlier races (2018, 2019, 2021-25; 2020 was cancelled), all from earlier seasons, none from 2026.
  - Raw mean Spearman 0.656 vs a global weighted 0.748.
  - Shrunk with k=4: **s_c = 0.705 vs global 0.748**, so at gamma=3 the grid multiplier is **0.88**, i.e. a slightly flatter grid dependence.
  - At k=2, gamma=3 the multiplier is 0.84.
  - This points the same way as T-010's note that COTA's grid-to-finish correlation is below average (0.52 vs 0.65), but more weakly.
- **US GP pre-quali top 5** (off vs on at gamma=3, k=4): the order is unchanged and the effect is tiny. This is illustrative only, since the term is not kept.

| Driver | P(win) off | P(win) on | P(podium) off | P(podium) on |
|---|---|---|---|---|
| ANT | 26.3% | 26.3% | 57.2% | 57.1% |
| HAM | 15.7% | 15.6% | 48.3% | 47.7% |
| RUS | 14.1% | 13.6% | 37.9% | 37.0% |
| LEC | 13.0% | 12.9% | 40.1% | 39.6% |
| VER | 10.8% | 11.0% | 35.6% | 35.9% |

The official prediction stays as in T-011 (the "off" columns).

## Files changed
- `backend/app/core/predictor.py`: `track_stickiness`, `_race_rho`, `_rho_table`, `circuit_of`, `by_race_table`; config fields; `prepare_history(track_results=, schedule=)`; the stickiness column in `race_features`; the multiplier in the level-2 fit and in `simulate`; `predict_race` attaches `attrs["stickiness"]`.
- `predict_race.py`: `--by-race`, `--stick-gamma`, loads `results_2018_2021.csv` and the schedule, and prints the stickiness line when enabled.
- `backend/tests/test_predictor.py`: 9 new test cases:
  - off equals baseline
  - shrinkage and a neutral new circuit
  - sign and ordering
  - **leakage**: scrambled target and future races, rebuilt from raw tables, give identical stickiness and probabilities in both modes
  - strictly-earlier races only
  - sum coherence in both modes
  - the by-race columns
- Data: `data/jolpica/derived/results_2018_2021.csv`, `data/jolpica/predictions/walk_forward_2026_by_race.csv`, plus the Jolpica raw cache for 2018-21.
- Not touched: `validate_scorecard.py`, `test_scorecard.py`, `BOARD.md`.

## Verification
`pytest backend/tests/test_predictor.py backend/tests/test_t012_qa.py` gives 47 passed (was 38). The default-config output is unchanged, because `stick_gamma=0` makes every multiplier exactly 1 (a test asserts this), and the by-race table reproduces the T-011 metrics.

## Open questions / risks
- **Noisy per-race signal:** a better estimator could use a per-circuit PL grid coefficient or net position changes, or lap data to exclude race-specific events (safety cars, rain). Not tried, to stay within budget.
- **Regulation change:** pre-2026 circuit history may not transfer, as the Monaco R6 gap suggests. The 0.5 weight is untuned because it can't be tuned on 2025 data.
- **Revisit** after about 25 races of 2026, when the circuit table contains 2026 data.
