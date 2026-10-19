# T-011 — data-scientist — Race-outcome predictor (win / podium / top-10)

**Status:** done
**Next owner:** qa-engineer, for T-012: verify leakage, reproduce the baseline comparison, check that probabilities are coherent. After QA, data-scientist does the T-013 Saturday rerun (instructions below).

## What I did
- **Data (Jolpica only):** results, qualifying, sprint and schedule for 2022-2026, cached as raw JSON under `data/jolpica/raw/`. A cold pull was about 120 requests at 1 req/s or slower. Only non-empty responses are cached, so the US GP qualifying is picked up automatically once Jolpica publishes it. Tidy tables are in `data/jolpica/derived/{results,qualifying,sprint,schedule}.csv`, and every row carries `season`.
- **Model** (`backend/app/core/predictor.py`; the docstring records why each choice was made):
  1. **Level 1:** Plackett-Luce (PL) ratings, one from race finishing orders (`theta_race`) and one from qualifying orders (`theta_quali`). Each rating is a team effect plus a driver-within-team effect with ridge shrinkage and exponential recency weights (half-life 8 races). They are fitted only on earlier rounds of the same season, and DNFs are excluded from the orders.
  2. **Level 2:** a PL outcome model, `u = b_race*theta_race + b_grid*(-log grid)`, fitted on earlier 2026 races (plus 2025 races as a coefficient prior). Both PL levels use truncated rankings (top 10 for the ratings, top 3 for the outcome model). The full-order fit made the front row far too unconfident: first pass gave P(win | pole) 0.44 vs 0.69 observed.
  3. **DNF component:** P(not classified) uses beta-binomial shrinkage, driver → team → 2026 field rate → 15% prior.
  4. **Pre-quali mode:** each simulation draws a qualifying order from PL(`a_quali*theta_quali`) as the expected-grid component, then runs the race model.
  5. **Monte Carlo:** 20k finishing orders (utility + Gumbel noise, DNFs placed last). Win probabilities sum to 1, podium to 3 and top 10 to 10 exactly.
  6. **Blend:** final = 0.75 × PL model + 0.25 × a rank-frequency table. The table is the grid-rank table in grid mode and the championship-rank table pre-quali.
- **Hyperparameters were tuned on the 2025 season walk-forward**, not on 2026. The 2026 evaluation was run after they were fixed:
  - Grid searched: half-life {2,4,8,inf} × lambda_team {.05,.2,1,3} × lambda_driver {.2,.5,2,8} × top-k {all,10} / {3,5,10,all}, plus blend weight {0,.25,.5,.75,1}.
  - The results form a broad plateau. Half-life 8 and inf tie on 2025 (Δ 0.003). I pre-committed to 8 because the 2026 pecking order shifted mid-season.
- **2025 data** is used only as down-weighted priors, and each prior was tested against 2026-only:
  - (a) Level-2 coefficient prior: kept (helps both modes).
  - (b) Driver-skill prior (2025 within-team effect): dropped (mixed).
  - 2025 team/car pace is never used.
  - A COTA-specific term was not added. T-010 rates driver-at-COTA history as mostly car-driven, and a single-track effect can't be validated walk-forward on 2026.
- **Practice long-run pace (orchestrator FYI):** I added `backend/app/core/openf1_practice.py`.
  - It pulls the OpenF1 FP laps and stints for all 17 2026 weekends, cached in `data/openf1_predictor/raw/` (separate from T-009's `data/openf1/`; 3.6 MB).
  - It computes a team long-run gap with a crude per-compound correction. The gap is wired in as an optional third level-2 feature (`use_fp`, `--fp`).
  - **It did not help** in the 2026 walk-forward: all metrics are slightly worse, and the fitted `b_fp` is about -0.06 to -0.12, i.e. the wrong sign. It is **off** by default. Raw Spearman between the FP feature and the finishing order averages only 0.19 per race and swings from -0.37 to 0.67.

## Files changed
- `backend/app/core/jolpica_client.py`: new. Cached and rate-limited Jolpica client plus flatteners and `load_all()`.
- `backend/app/core/predictor.py`: new. Ratings, outcome model, DNF model, simulation, rank baselines, walk-forward, scoring and calibration.
- `backend/app/core/openf1_practice.py`: new. OpenF1 practice long-run gaps (optional feature, off).
- `predict_race.py`: new, at the repo root. Prints the prediction table (`--mode auto|prequali|grid`, `--grid-file`, `--offline`, `--fp`) and runs `--evaluate`.
- `backend/tests/test_predictor.py`: new, 28 tests:
  - PL gradient checks
  - simulation coherence
  - DNF shrinkage
  - leakage (scrambles the target's and later races' results, positions, status, points and quali; output must be unchanged)
  - baseline leakage
  - the blend
  - the FP feature
  - Jolpica cache behaviour (empty responses not cached; offline mode)
  - the 2025 coefficient prior is never applied to a 2025 target (`_training_races` guard)
  - a real-data smoke test
- `requirements.txt`: pinned `scipy==1.17.1` and `requests==2.33.0`. Both were already installed as fastf1 dependencies and are now imported directly.
- `data/jolpica/{raw,derived,eval,predictions}/` and `data/openf1_predictor/`: new data directories.
- I did not touch `data_loader.py`, `tire_model.py`, `validate_replay.py`, `pull_races.py`, `data/processed/` or `cache/`.

## Verification
- `pytest backend/tests/test_predictor.py`: **28 passed**.
- Full `pytest`: 209 passed, 2 failed. Both failures are in `backend/tests/test_scorecard.py` (T-003's in-progress `validate_scorecard.py`), and its tests use their own tmp fixtures, not my files. See the note to T-003 below.
- `python predict_race.py --evaluate --fp --sims 10000` saves to `data/jolpica/eval/walk_forward_2026.csv` and `_predictions.csv`.
- I did not run `validate_replay.py`: the simulator and tyre model are untouched.

### Walk-forward 2026, rounds 2-17 (16 races; each race predicted only from earlier rounds; lower is better)

Log-loss and Brier are per-driver binary scores averaged per race. `winner NLL` = -log P(actual winner).

| method | win LL | win Brier | podium LL | podium Brier | top10 LL | winner NLL |
|---|---|---|---|---|---|---|
| uniform | .1849 | .0434 | .3983 | .1178 | .6890 | 3.09 |
| **baseline: grid order** | .0937 | **.0230** | .2088 | **.0650** | .5161 | 1.38 |
| **baseline: championship order** | .1176 | .0357 | .2505 | .0819 | .5359 | 1.70 |
| baseline: recent form (avg finish, last 3) | .1584 | .0360 | .3576 | .1015 | .5764 | 2.60 |
| PL pre-quali, 2026-only | .1183 | .0339 | .2627 | .0858 | .5141 | 1.76 |
| PL pre-quali, +2025 coef prior | .1135 | .0335 | .2472 | .0821 | .5031 | 1.65 |
| PL grid, 2026-only | .0827 | .0230 | .2189 | .0673 | .4854 | 1.14 |
| PL grid, +2025 coef prior | **.0773** | **.0207** | .2112 | .0676 | .4844 | **1.08** |
| PL grid, +2025 driver prior (dropped) | .0813 | .0224 | .2195 | .0675 | .4885 | 1.12 |
| **chosen pre-quali** (blend + coef prior) | **.1133** | **.0337** | **.2456** | **.0813** | .5041 | **1.64** |
| **chosen grid** (blend + coef prior) | .0790 | .0211 | **.2075** | .0664 | **.4839** | 1.10 |
| chosen grid + FP long runs (not kept) | .0798 | .0212 | .2094 | .0670 | .4860 | 1.12 |

Rounds 6-17 (12 races, less early-season noise) tell the same story:
- chosen grid: .0877 / .2002 / .4753 (win / podium / top10 LL)
- grid baseline: .1022 / .2021 / .5090
- chosen pre-quali: .1268 / .2443 / .4876
- championship baseline: .1270 / .2456 / .5218

**How the model did vs the baselines:**
- **With the grid:** the model beats grid order on win log-loss (-16%), winner NLL (1.10 vs 1.38), podium log-loss (slightly: .2075 vs .2088) and top-10. **It is slightly worse on podium Brier** (.0664 vs .0650).
  - The pure PL model loses to grid order on podium (.2112 vs .2088). That is why it is blended with the grid table. The blend weight was fixed on 2025, and the blend beats both of its parts on win and podium log-loss.
- **Pre-quali:** the model beats championship order on every metric, but the margin is **small**: win LL .1133 vs .1176, podium .2456 vs .2505, top-10 .5041 vs .5359.
  - On rounds 6-17 it essentially **ties** championship order on win (.1268 vs .1270).
  - Honest summary: before qualifying, the championship table already holds most of the signal. The model adds coherent DNF-aware top-10 probabilities and a small edge.
- Recent-form average is the worst baseline.
- **Calibration** (pooled, rounds 2-17, chosen grid model):

  | Event | Predicted bin | Mean predicted | Observed | Events (pred → obs) |
  |---|---|---|---|---|
  | Win | 0.35-0.65 | 0.54 | 0.64 | 7.5 → 9 |
  | Win | 0.15-0.35 | 0.22 | 0.21 | — |
  | Win | 0.05-0.15 | 0.09 | 0.05 | — |
  | Podium | 0.65-1 | 0.78 | 0.76 | — |
  | Podium | 0.35-0.65 | 0.48 | 0.40 | — |
  | Podium | 0.15-0.35 | 0.24 | 0.32 | — |

  - Mean P(win | P1 start): 0.56 for the chosen model vs 0.54 for the grid baseline, against 0.69 observed (n=16). It is mildly under-confident on pole and inside T-010's 35-60% sanity range. Pooled 2022-26 pole conversion is 58%.
  - Pre-quali podium: 0.25 predicted vs 0.33 observed in the 0.15-0.35 bin, and 0.48 vs 0.41 in the 0.35-0.65 bin.
  - n is small in every bin (14-60 driver-races).
- **2025 tuning season** (walk-forward rounds 2-24, chosen top-k and blend w=0.25, no 2025 coefficient prior because that would be the season itself): grid model win/podium LL .0880/.1941 vs grid baseline .0930/.2014; pre-quali .1334/.2519 vs championship .1584/.2758.
- **Sensitivity to half-life** (2026 walk-forward and today's table): the metrics barely move. Pre-quali win LL is .1129 / .1133 / .1139 for half-life 4 / 8 / inf. Verstappen's US GP P(win) moves from 8% to 13% across that range, because his surge is only two races old.

### US GP 2026 pre-quali prediction (generated 2026-10-19 15:33; saved to `data/jolpica/predictions/2026_R18_prequali_20261019_1533.csv`)

| # | Driver | E[grid] | P(win) | P(podium) | P(top 10) | P(DNF) |
|---|---|---|---|---|---|---|
| 1 | ANT (Mercedes) | 4.3 | 26.3% | 57.2% | 94.2% | 4.8% |
| 2 | HAM (Ferrari) | 4.0 | 15.7% | 48.3% | 89.8% | 8.7% |
| 3 | RUS (Mercedes) | 4.3 | 14.1% | 37.9% | 86.3% | 10.2% |
| 4 | LEC (Ferrari) | 4.3 | 13.0% | 40.1% | 84.9% | 11.4% |
| 5 | VER (Red Bull) | 5.1 | 10.8% | 35.6% | 84.2% | 16.6% |
| 6 | NOR (McLaren) | 4.7 | 9.1% | 29.3% | 77.6% | 21.7% |
| 7 | PIA (McLaren) | 5.6 | 4.4% | 18.9% | 74.8% | 19.0% |
| 8 | HAD (Red Bull) | 7.9 | 2.8% | 12.8% | 73.2% | 15.1% |

Sums: win 1.000, podium 3.000, top-10 10.000. Monte Carlo noise at 20k sims is about ±0.3 percentage points. Pre-quali probabilities are deliberately flat: the grid is the single strongest input, and P(win) for a pole-sitter is about 55%.

## T-013: Saturday 2026-10-24 rerun (exact steps)
1. **Qualifying** runs Sat 2026-10-24 16:00 CDT (21:00 UTC) and ends around 22:00 UTC.
   - **When Jolpica publishes it is unverified.** Neither T-010 nor I could measure the lag. Jolpica is volunteer-maintained and results usually appear within a few hours of a session, sometimes later.
   - To check whether it's up: `curl "https://api.jolpi.ca/ergast/f1/2026/18/qualifying.json"`. It's up when `"total"` is > 0 and there are 22 rows.
2. Run `python predict_race.py`. Auto mode switches to grid mode as soon as Jolpica has round-18 qualifying.
   - The header should say `qualifying for this round AVAILABLE` and `grid = Jolpica qualifying order (NO penalties applied)`.
   - Nothing else is needed: earlier rounds are served from cache, and empty responses are never cached.
3. **Grid penalties:** Jolpica's qualifying endpoint is the quali classification, not the grid. Post-penalty grid only appears in Jolpica race results, after the race.
   - If the FIA announces any penalty or pit-lane start, write `grid.csv` with columns `code,grid` (22 rows, three-letter codes as in the table, pit-lane start = 0).
   - Then run `python predict_race.py --mode grid --grid-file grid.csv`. The grid file also defines the entry list, so a reserve driver substitution goes in here too. Their code must exist in Jolpica 2026 entry lists, otherwise the script stops with a clear message.
4. **If Jolpica is late:** build `grid.csv` from the official FIA/F1 classification and use step 3. The model doesn't need Jolpica's round-18 data at all, only the grid.
5. Optional on Friday evening: `python predict_race.py --mode prequali` for an update after the Friday information. Note that Jolpica has no practice data, so this only changes if the entry list changes.
6. **FP long runs:** `--fp` pulls this weekend's OpenF1 practice long runs. They didn't help in walk-forward, so **don't use them for the headline number**. At most, report them as a side note.
7. **Output:** a table on stdout and a CSV in `data/jolpica/predictions/2026_R18_grid_<timestamp>.csv`. The CSV includes `p_*_model` and `p_*_baseline` components.

## Interface / contract changes
- No changes to existing modules or APIs. New public functions:
  - `predictor.predict_race(hist, season, rnd, mode, cfg, entry_list=None, grid=None) -> DataFrame` with columns `p_win`, `p_podium`, `p_top10`, `p_dnf`, `exp_pos` (and `exp_grid` in pre-quali mode).
  - `predictor.walk_forward(...)`, `score`, `calibration_table`.
  - `jolpica_client.load_all(...)`.
- Jolpica cache layout:
  - Completed seasons are season-level files: `data/jolpica/raw/<season>_results__o<offset>.json`, `<season>_qualifying__o<offset>.json`, `<season>_sprint__o<offset>.json`.
  - 2026 is per round: `2026_<round>_results__o0.json`, `2026_<round>_qualifying__o0.json`, `2026_<round>_sprint__o0.json`.
  - The schedule is `<season>__o0.json`.

## Requests for other roles
- **qa-engineer (T-012):**
  1. Leakage: review `_past_orders`, `dnf_probabilities`, `_training_races` and `RankBaseline.table`, all of which should filter `round < r`. `test_no_leakage_from_target_or_future_races` and `test_baselines_no_leakage_and_coherent` are the automated checks.
  2. Reproduce the table with `python predict_race.py --evaluate --offline --fp --sims 10000` (about 40 s).
  3. Sums are 1/3/10 per race.
  4. Hyperparameters were not fitted on 2026, except two binary keep/drop decisions on 2025 priors and the FP feature, which the task explicitly asked to decide by 2026 walk-forward.
  5. The baselines get 2022-25 history the model doesn't use, which is generous to them.
- **orchestrator:**
  - New `scipy` and `requests` pins in `requirements.txt`.
  - Decide whether `data/jolpica/raw` (3.1 MB) and `data/openf1_predictor/raw` (3.6 MB) get committed. Committing them makes the T-013 rerun independent of API availability for past rounds.
- **data-scientist (T-003 / scorecard owner):** `validate_scorecard.load_jolpica_qualifying` reads `data/jolpica/raw/2026_qualifying__o0.json`. This client never writes that file for 2026 (per-round files instead; see the cache layout above). Read per-round files, or use `jolpica_client.qualifying(2026, round, offline=True)`.

## Open questions / risks
- **Small samples:** 16 evaluated races. The pre-quali margin over championship order is within noise, while the grid-mode win/winner-NLL gain is more substantial.
- **DNFs** are independent per car with no SC/weather link. Two cars from the same team can't fail together, and lap-1 incidents aren't correlated across cars.
- **Not modelled:** rain (Austin forecast unknown), team orders around Antonelli's possible title clinch, grid-penalty risk in pre-quali mode (the simulated grid is the quali order), and COTA-specific effects. COTA's grid-to-finish correlation is a bit lower than average (T-010: 0.52 vs 0.65), which argues for slightly flatter probabilities than shown.
- **FP long-run follow-up:** the extraction is crude, with no fuel-load correction and FP1 reserve drivers only filtered via the entry list. A better version could use FP2 only, compound offsets from `tire_model`, and lap-count-weighted shrinkage. It should be re-tested walk-forward before use.
- `exp_pos` and `p_dnf` come from the PL model only (not blended).

## QA fixes (T-012 findings 1 and 2)
- `predict_race.py::_entry_and_grid` (grid/auto): entry list is now the union of Jolpica qualifying rows and the previous round's race entrants. Drivers with no quali row start after all qualifiers, in previous-round entry order. Prints a WARNING listing them, and a WARNING if the field is not 22.
- `--grid-file`: SystemExit on duplicate codes or duplicate non-zero grid positions; WARNING if fewer than 22 rows.
- `test_t012_qa.py`: xfail removed, now asserts 22 cars incl. VER/STR and a warning. `pytest test_predictor.py test_t012_qa.py`: 38 passed.
- Walk-forward metrics unchanged: `walk_forward` builds entrants from `predictor.py` (actual race results), not from `_entry_and_grid`, so the fix touches only the live CLI path. `predictor.py` was not edited.
