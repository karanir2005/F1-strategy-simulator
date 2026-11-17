# T-057 — data-scientist — Predictor probability calibration check

**Status:** done
**Next owner:** orchestrator — close task; no code follow-up needed.

## What I did
- Wrote `data/predict_eval/calibration_check.py`: walk-forward grid-mode predictions (in-memory via `predictor.walk_forward`, not `--by-race`), reliability tables with Wilson intervals, Brier / log loss / ECE vs the grid-order baseline (`RankBaseline("grid")`), paired race-bootstrap on Brier difference, walk-forward logit-temperature (and power-renormalised for p_win) recalibration fitted only on earlier races, plus a 2025->2026 cross-season T check.
- Ran it for 2026 (20 races) and 2025 (24 races; cheap, whole run ~25 s).
- Findings: `team/research/T-057-calibration.md`.

## Key numbers
- Verdict: roughly fine. 2026 mildly underconfident for favourites (favourite mean P(win) 0.567, won 0.70, n=20), outsiders in 0.10-0.20 bin overconfident (0.144 pred vs 0.048 obs, n=21). 2025 well calibrated (favourite 0.604 vs 0.625).
- Model vs grid baseline, 2026 R2-20 Brier / log loss: win 0.0212/0.0758 vs 0.0238/0.0982; podium 0.0650/0.2058 vs 0.0672/0.2214; top10 0.1519/0.4718 vs 0.1621/0.5085. Only top10 Brier diff CI excludes 0. ECE 0.018 / 0.031 / 0.034 (grid baseline similar or slightly lower).
- Walk-forward temperature correction hurts held-out in both seasons (2026 win Brier 0.0212 -> 0.0223, log loss 0.0758 -> 0.0794; 2025 win 0.0264 -> 0.0277). Cross-season 2025-fitted T moves 2026 Brier by <0.0002.
- No config flag added; shipped default unchanged.

## Files changed
- `data/predict_eval/calibration_check.py` — new script
- `data/predict_eval/out/*.csv` — reliability tables, summary, model walk-forward predictions for 2025/2026
- `team/research/T-057-calibration.md` — findings

## Verification
- 2026 model predictions from my run match shipped `data/jolpica/predictions/walk_forward_2026_drivers.csv` (440 rows, max |dp_win| 1e-4).
- md5 of the four default artifacts in `data/jolpica/predictions/` and `data/jolpica/eval/` checked before and after: unchanged.
- Not run: pytest (no product code touched). Not verified: calibration of prequali mode, or with FP features.

## Interface / contract changes
none

## Requests for other roles
- none

## Open questions / risks
- 20 (2026) and 24 (2025) races is a small sample; the p_win T of 0.83 in 2026 is not detectable out of sample. Revisit with 2027 data pooled.
- Wilson intervals treat the 22 drivers of a race as independent, so they are optimistic.
