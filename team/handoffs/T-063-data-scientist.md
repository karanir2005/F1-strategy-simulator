# T-063 — data-scientist — Predictor calibration under strict walk-forward

**Status:** done (negative result, no default changed)
**Next owner:** orchestrator — close task.

## What I did
- Tried (a) in-model knobs (new off-by-default `PredictorConfig.utility_scale`, blend weights, stick_gamma, topk_level2, w2025), (b) pooled/own-season shrunk Platt recalibration, (c) slope vs slope+intercept, walk-forward-chosen utility scale, (d) outsider-bin audit. Script `data/predict_eval/t063_experiments.py`; results `data/predict_eval/out/t063_A.csv`, `t063_B.csv`; write-up in the T-063 section of `team/research/T-057-calibration.md`.

## Key numbers (held-out win log loss diff vs default, 95% race-bootstrap)
- Best: utility_scale 1.15: 2026 -0.0024 [-0.0054,+0.0008], 2025 -0.0003 [-0.0037,+0.0032]. Pooled Platt (lam 5): 2026 -0.0019 [-0.0047,+0.0011], 2025 +0.0032 (worse). Walk-forward-chosen scale: 2026 -0.0015, 2025 +0.0006..+0.0031.
- Hit rates unchanged by every scale (winners 14/20, 15/24; podium 36/60, 57/72 by argmax / top-3 p_podium).
- The 0.25 blend is not the cause for p_win (grid-mode win blend is already 0); podium/top10 blend optimum flips between seasons.
- Outsider bin: 19/21 were P1-P3 starters ranked 2nd-4th favourite, not back-starters (so not T-052). Same pattern flips sign in 2025 (0.153 pred vs 0.333 obs), so it is a season dominance regime, not a fixable bias.
- Nothing clears the ship bar: no interval excludes 0 in 2026 and 2025 is neutral or worse.

## Files changed
- `backend/app/core/predictor.py`: added `utility_scale: float = 1.0` to PredictorConfig and 4 lines in `predict_race` applying it. Default 1.0 is a no-op; no signature changes.
- `data/predict_eval/t063_experiments.py`, `data/predict_eval/out/t063_{A,B}.csv`, `team/research/T-057-calibration.md`.

## Verification
- Default behaviour unchanged (scale 1.0 skips the code path; default row in the sweep equals T-057 numbers). Shipped artifacts not regenerated, since defaults unchanged. pytest run: see orchestrator note below.

## Interface / contract changes
none (optional new config field, default off)

## Open questions / risks
- 20 winners in 2026; the sharpen-by-~1.15 signal is consistent in sign but not significant. Revisit with 2027 data.
