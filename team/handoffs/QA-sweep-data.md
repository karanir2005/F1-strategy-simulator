# QA-sweep — data-scientist — predictor / quali / tyre+SC review before Austin (R18)

VERDICT: PASS WITH NOTES. No leakage, no data errors that change the R18 prediction. Several calibration issues and cheap gains are listed below. Review only: no code or data changed. pytest: **344 passed**.
Scratch scripts and the walk-forward dump: `C:\Users\rushi\AppData\Local\Temp\claude\C--Users-rushi-Desktop-Folders-Code-F1-strategy-simulator\bbb5655a-bfbb-4671-b4b7-bfaa13ddf821\scratchpad\` (wf.py, calib.py, var.py, qgrid*.py, wf_all.csv).

## Data (data/jolpica/derived/*.csv): clean
- No duplicate (season, round, driver) rows in results, qualifying, sprint or results_2018_2021. There are no season overlaps between results and results_2018_2021, and `session` is only `race` in results (sprint rows are kept separately).
- circuit_ids match between results, quali and schedule. Austin is `americas` in all 7 earlier races (2018, 2019, 2021-25). Schedule R18 = americas, `has_sprint=False`, quali 2026-10-24.
- The R18 entry list (`entrants_for`, from R17) has 22 drivers, 11 teams x 2, with no code collisions. Lawson shows up at red_bull for R12-14, with Tsunoda at rb and Hadjar missing. That looks like a real substitution (Hadjar is back from R15) but please confirm it once.
- Minor: Jolpica gives a duplicate `quali_position` in 2023 R10, 2024 R8/R15/R17 and 2025 R7 (e.g. 2024 Monaco ALO=HUL=14). This only touches the 2025 a_quali prior and the baseline tables. Negligible.
- **SC data bug:** `data/staging/Belgium_2025_*` marks laps 1-4 as green. Those laps were run behind the safety car after the red flag (median 172 s against 122 s green), and the track_status file shows only Red then AllClear. As a result `safety_car_params.json` misses 1 SC event (26/40 should be 27/40). Tyre fits are unaffected because Belgium_2025 is in `EXCLUDED_2025_RACES` (tire_model.py:66). The fix belongs in openf1_loader/calibrate_safety_cars: flag slow laps that follow a red flag. Austin's own SC history is fine.

## Leakage: none found
Every fit filters to strictly earlier races (`round < r` or an earlier season). That covers stickiness (predictor.py:284), DNF (607), the baselines (833-837) and the quali tables (978-981). Re-running walk-forward reproduces T-037 exactly (2026 grid win LL 0.0758, 2025 0.0847). One soft point on selection: `w2025_level2` (predictor.py:117) and the T-037 ship decision both used the 2026 results as their acceptance gate, so the 2026 numbers are no longer a clean holdout.

## Calibration (walk-forward 2025 R2-24 + 2026 R2-17, 39 races, shipped config)
| mode | bin | mean pred | observed | n |
|---|---|---|---|---|
| grid win | 0.35-0.55 / 0.55-0.75 / >0.75 | .48 / .63 / .78 | .60 / .67 / 1.00 | 15 / 15 / 6 |
| grid win | 0.02-0.05 | .033 | .000 | 55 |
| prequali win | 0.35-0.55 | .43 | .28 | 25 |
| prequali win | 0.05-0.15 | .088 | .126 | 103 |
| pole (T-035) | top pick | .39 | .26 | 39 |

- **Grid mode is under-confident:** the top pick averages 0.574 but wins 0.667 of the time.
- **Pre-quali is over-confident on the favourite:** the top pick averages 0.41 but wins 0.31.
- **The pole model is over-confident on its favourite too:** p_pole 0.35-0.55 comes in at 0.29.

## T-037 stickiness: helps grid-mode win only
Compared with gamma 0, gamma 4 makes **top-10 LL worse in both seasons** (2025 0.4882->0.4989, 2026 0.4839->0.4877), makes 2025 podium worse (0.1941->0.1966), and makes **pre-quali winner NLL worse in both** (2025 1.752->1.771, 2026 1.641->1.670). The gain comes from 9 configs tried on 23 races, using a 0/1 signal.

## What could make tomorrow's Austin call wrong
1. **The pre-quali favourite is mostly the blend, not the model.** ANT gets 26.1% overall but 19.2% from the model. The championship-rank baseline gives the points leader p_win 0.47 (predictor.py:747-755). Leaders won 0.57/0.86/0.35/0.22 in 2022-25 and 6/16 in 2026, and ANT hasn't won since R14. VER has won R16 and R17 but gets only 10.9% (13.3% from the model, 3.8% from the baseline).
2. **Penalties.** Grid mode with no override uses the quali order (predict_race.py:141-146), but walk-forward was scored on the real post-penalty grid. Using the quali order as the grid actually **scored better** in 2026 R2-17: win LL 0.0758->0.0726, top-10 LL 0.4877->0.4724. The top-10 gain is spread over R10, R15, R16 and R17 (penalised fast cars recover); the win gain comes entirely from R13 (ANT won from P19 after qualifying P7). So -log(grid) under-rates penalised fast cars. If a front-runner takes a power-unit penalty on Sunday, the override grid will under-rate him.
3. **McLaren DNF risk is inflated:** p_dnf is 21.7% for NOR and 19.0% for PIA. `dnf_probabilities` (predictor.py:603-622) counts DNS as DNF (PIA R1+R2, NOR R2) and has no recency weighting. The field rate fell from 22.7% in R1-8 to 15.7% in R9-17.

## Cheap improvements (ranked by expected gain)
1. **Stop blending the baseline into p_win in grid mode** (keep w=0.25 for podium/top-10). Measured: win LL 2025 0.0847->0.0838 and 2026 0.0758->0.0747; winner NLL 1.015->0.994 and 1.032->1.017. Bigger alternative: temperature T=1.25 on p_win, fitted on 2025 and tested on 2026, gives win LL 0.0758->**0.0706** and NLL 1.032->**0.956**. Both seasons agree.
2. **Add -log(quali_pos) as a 3rd level-2 feature next to -log(grid)** (one extra coefficient). Evidence from point 2 above: top-10 LL -0.015 in 2026 using the quali order alone. Untested as a joint feature; needs a 2025-tuned walk-forward.
3. **Make stickiness grid-mode-only** (gamma 0 pre-quali): pre-quali win LL 2025 0.1342->0.1334, 2026 0.1149->0.1133. Also drop DNS from the DNF counts or add recency weighting to them (untested; affects NOR/PIA podium odds).

None of the three changes the R18 pre-quali pick (model-only ANT 0.192 > HAM 0.169). They change the probabilities shown.
