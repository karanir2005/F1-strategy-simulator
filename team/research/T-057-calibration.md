# T-057 — Calibration check of the race predictor (2026-11-16)

## Summary

**The model is roughly fine; if anything it is mildly underconfident about the favourites in 2026, and well calibrated in 2025.** Nothing here is strong enough to act on. A one-parameter recalibration fitted walk-forward (only on earlier races) made held-out Brier and log loss *worse* in both seasons, so no config flag was added and the shipped default is unchanged.

Treat all of this as weak evidence. 2026 is 20 races (19 scored against the grid baseline), which is 20 winners, 60 podium places and about 200 top-10 places. The 22 drivers in one race are not independent (exactly one wins), so the Wilson intervals below are optimistic. A bin with 3-10 driver-races tells you almost nothing.

Setup: grid mode with the shipped `PredictorConfig`, in-memory `predictor.walk_forward` (each race predicted from earlier rounds plus its own grid). 2026 predictions match the shipped `walk_forward_2026_drivers.csv` to 1e-4. Metrics pool driver-races (per-driver binary Brier / log loss). Model-vs-baseline uses R2-R20 (2026) and R2-R24 (2025), because the grid baseline starts at R2. The reliability tables use R1-end. The baseline is `predictor.RankBaseline("grid")`, a frequency table of P(event | grid slot) built from 2022-2025, so it is generous to the baseline.

## Scores (model vs grid-order baseline; lower is better)

| season | event | Brier model | Brier grid | Brier diff 95% (paired race bootstrap) | log loss model | log loss grid | ECE model | ECE grid |
|---|---|---|---|---|---|---|---|---|
| 2026 | win | 0.0212 | 0.0238 | [-0.0051, +0.0001] | 0.0758 | 0.0982 | 0.018 | 0.015 |
| 2026 | podium | 0.0650 | 0.0672 | [-0.0069, +0.0026] | 0.2058 | 0.2214 | 0.031 | 0.019 |
| 2026 | top10 | 0.1519 | 0.1621 | [-0.0213, -0.0009] | 0.4718 | 0.5085 | 0.034 | 0.029 |
| 2025 | win | 0.0264 | 0.0279 | [-0.0060, +0.0034] | 0.0840 | 0.0931 | 0.018 | 0.015 |
| 2025 | podium | 0.0567 | 0.0584 | [-0.0067, +0.0039] | 0.1964 | 0.2013 | 0.021 | 0.019 |
| 2025 | top10 | 0.1606 | 0.1613 | [-0.0101, +0.0093] | 0.4940 | 0.4951 | 0.027 | 0.044 |

- The model beats the grid baseline on every Brier and log-loss number, but only the 2026 top-10 Brier interval excludes zero. Log loss gaps are larger than Brier gaps because the grid baseline puts too little mass on the true winner when the favourite is not on pole.
- ECE (count-weighted gap between predicted and observed per bin) is 2-3 percentage points for both model and baseline. The grid baseline is slightly better on ECE in 5 of 6 rows. With ~8 bins and tiny counts the ECE of either is mostly noise; the difference is not meaningful.
- Pooled mean probability equals the base rate by construction (probabilities sum to 1/3/10 per race), so only the *shape* can be mis-calibrated, and that is what the tables below test.

## Reliability, 2026 (model, R1-R20)

p_win: predictions under 0.20 are slightly too high; above 0.20 too low.

| bin | n | mean pred | observed | Wilson 95% |
|---|---|---|---|---|
| <0.02 | 330 | 0.003 | 0.000 | 0-0.012 |
| 0.02-0.05 | 35 | 0.033 | 0.029 | 0.005-0.145 |
| 0.05-0.10 | 28 | 0.068 | 0.036 | 0.006-0.177 |
| 0.10-0.20 | 21 | 0.144 | 0.048 | 0.008-0.227 |
| 0.20-0.35 | 7 | 0.238 | 0.429 | 0.158-0.750 |
| 0.35-0.50 | 6 | 0.440 | 0.667 | 0.300-0.903 |
| 0.50-0.70 | 10 | 0.600 | 0.700 | 0.397-0.892 |
| 0.70+ | 3 | 0.832 | 1.000 | 0.438-1.000 |

The race favourite averaged P(win) 0.567 and won 14 of 20 (0.70). That is underconfidence at the top in 2026, a Mercedes-dominated season, and the same direction as the "P(win|pole) 0.44 vs 0.69" problem the predictor's docstring says was already partly fixed by `topk_level2=3`. The 0.10-0.20 bin (21 driver-races, 1 win) is the opposite: overconfident about outsiders. Together that is "too flat among the top few, then too much mass on the second tier". All intervals are wide; the 0.10-0.20 and 0.20-0.35 bins are the only ones whose interval excludes the predicted mean, and with 7-21 rows I would not lean on it.

p_podium (2026): near the diagonal, noisy. Only the 0.15-0.30 bin (0.22 predicted, 0.39 observed, n=33) is off, and its interval (0.25-0.56) just excludes the prediction. p_top10 (2026): bottom bin predicted 0.09, observed 0.04 (n=93), the model puts slightly too much top-10 mass on backmarkers; the rest sits on the diagonal.

## Reliability, 2025 (model, R1-R24)

Better behaved. Favourite mean P(win) 0.604 vs observed 0.625 (n=24). p_win bins are within noise except 0.10-0.20 (0.153 predicted, 0.333 observed, n=18) and the 0.70+ bin (0.82 vs 0.67, n=9). Podium and top-10 bins track the diagonal; the top-10 bins at 0.95+ (0.955 vs 0.82, n=11) are too small to read. Full tables: `data/predict_eval/out/reliability_{2025,2026}_{win,podium,top10}.csv`.

## Recalibration test (walk-forward)

Correction tried: logit temperature `p' = sigmoid(logit(p)/T)` (T<1 sharpens, T>1 flattens), per event; and for p_win a power version `p^a / sum(p^a)`. T / a were fitted by minimising log loss on rounds strictly before the target round (identity until two earlier races exist).

| season | event | Brier model -> temp | log loss model -> temp | T fitted on the full season (in-sample, not usable) |
|---|---|---|---|---|
| 2026 | win | 0.0212 -> 0.0223 | 0.0758 -> 0.0794 | 0.83 |
| 2026 | podium | 0.0650 -> 0.0653 | 0.2058 -> 0.2092 | 0.99 |
| 2026 | top10 | 0.1519 -> 0.1539 | 0.4718 -> 0.4777 | 1.03 |
| 2025 | win | 0.0264 -> 0.0277 | 0.0840 -> 0.0945 | 0.94 |
| 2025 | podium | 0.0567 -> 0.0582 | 0.1964 -> 0.2083 | 0.98 |
| 2025 | top10 | 0.1606 -> 0.1622 | 0.4940 -> 0.5006 | 1.06 |

- Walk-forward correction is worse everywhere, including after round 6 when it has 5+ races to fit on. Power-renormalised p_win is no better (2026 Brier 0.0229, 2025 0.0280). The reason is the usual one: with 1-20 wins the fitted T is dominated by a few races and bounces around (2026 p_win T drifted 0.79-0.85 over the last five rounds, 2025 0.88-1.00).
- Cross-season: T fitted on all of 2025 (win 0.95, podium 0.97, top10 1.05) applied to 2026 changes Brier by under 0.0002 and log loss by -0.0009 (win), 0 (podium), 0 (top10). That is negligible: 2025 says the model is almost exactly calibrated, so there is nothing to transfer.
- Even the in-sample best T is within 0.06 of 1.0 for podium and top-10; only 2026 p_win (T=0.83) suggests any sharpening, and that one cannot be exploited walk-forward.

## Recommendation

1. Do not recalibrate. No config flag was added, and the shipped default is unchanged.
2. If the 2026 favourite pattern holds in 2027 (favourite wins well above its stated P), revisit with a larger sample: pool 2026 + 2027 and test the p_win sharpening (T ~0.85) walk-forward then. At 20 races it is not detectable.
3. For users reading the tables: p_win above ~0.2 is, if anything, conservative in 2026; p_win in 0.10-0.20 for non-favourites is somewhat generous. p_podium and p_top10 can be taken at face value.

## Reproduce

`.venv\Scripts\python data\predict_eval\calibration_check.py` (about 25 s, offline, cached Jolpica data). Writes only to `data/predict_eval/out/`. Default artifacts under `data/jolpica/` were checksummed before and after and are unchanged.
