# T-029 — backend-engineer — pre-quali delta fields, stickiness re-eval, track DNF

**Status:** done. Full pytest: 331 passed (5 new in `backend/tests/test_t029_backend.py`). No shipped default changed, no pins changed. Next: qa-engineer; frontend-engineer for the delta strip.

## 1. API: `drivers[].p_win_prequali`, `p_podium_prequali`
Grid mode only, `number|null` (0..1). From the newest `data/jolpica/predictions/2026_R{round}_prequali_*.csv` of the same round (newest by filename stamp). Null when no file, unreadable, or driver missing; never estimated. Absent in `pre_quali` mode. Stale fallback attaches them too. Contract text: `team/contracts/prediction-api.md` (Addendum T-029). Code: `prequali_snapshot()` + `shape()` in `backend/app/services/prediction_service.py`.

## 2. Track stickiness (`stick_gamma`): stays OFF
Walk-forward, grid mode, 4000 sims, lower is better. 2025 = R2-24 with the 2025 config (no prior seasons); 2026 = R2-17 shipped config. Gamma picked on 2025 only.
| season | gamma | win | podium | top10 |
|---|---|---|---|---|
| 2025 | 0 | 0.0882 | 0.1912 | 0.4880 |
| 2025 | 3 | 0.0885 | 0.1903 | 0.4872 |
| 2025 | 8 | 0.0895 | 0.1895 | 0.4861 |
| 2026 | 0 | 0.0796 | 0.2075 | 0.4870 |
| 2026 | 3 | 0.0782 | 0.2071 | 0.4869 |
| 2026 | 8 | 0.0777 | 0.2062 | 0.4870 |
(gamma 1, 2, 5, 12 are monotone in between.) Win log-loss gets worse with gamma on 2025 and better on 2026; podium improves on both but only slightly. Sign is inconsistent on win, so the rule fails. Prequali is worse or flat on both seasons. Left at 0.

## 3. Track-level DNF: implemented, OFF by default
`PredictorConfig.dnf_track_m` (0 = off), `dnf_track_clip` (0.9); `track_dnf_ratio()` in `predictor.py`; CLI `predict_race.py --dnf-track-m M`. p_dnf is multiplied by `(obs_c + m)/(exp_c + m)`: not-classified starts at the circuit in strictly earlier races (2018-26) over the same races' season-rate expectation, shrunk toward 1 by m pseudo-DNFs. Leakage-tested (target race ignored).
Grid mode (m picked on 2025):
| season | m | win | podium | top10 |
|---|---|---|---|---|
| 2025 | off | 0.0882 | 0.1912 | 0.4880 |
| 2025 | 1 | 0.0879 | 0.1925 | 0.4929 |
| 2025 | 4 | 0.0880 | 0.1922 | 0.4919 |
| 2025 | 16 | 0.0880 | 0.1917 | 0.4899 |
| 2026 | off | 0.0796 | 0.2075 | 0.4870 |
| 2026 | 1 | 0.0790 | 0.2055 | 0.4850 |
| 2026 | 4 | 0.0791 | 0.2058 | 0.4852 |
| 2026 | 16 | 0.0793 | 0.2064 | 0.4853 |
2026 improves on all three, 2025 improves win only and worsens podium and top10 at every m tried (more shrinkage shrinks both). Not consistent, so stays off. Worth revisiting after more seasons.

Not touched: frontend/, cache/, optimizer.py, race_simulator.py, monte_carlo.py, tire_model.py. Scratch scripts in the session scratchpad (`stick.py`, `dnf.py`); `walk_forward_2026*.csv` not regenerated.
