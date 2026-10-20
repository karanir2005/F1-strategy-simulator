# T-026 — backend-engineer — robust stint caps + noise/pit recalibration

**Status:** done. pytest 309 passed (was 308; +1 new optimizer test; schema test updated). **Next:** qa-engineer.

## 1. Robust stint caps
- `build_track_params.py`: new `typical_stint_laps` {compound: int|null} = p90 of observed stint lengths from the same stint set as `max_stint_laps` (2026 where raced, else 2025; other season only fills missing compounds); null if < 5 stints. `track_params.json` regenerated; diff is purely additive (+114 lines).
- `optimizer.optimize(..., typical_stint_laps=None)`: cap = min(max, typical) when typical is non-null, then the existing floor 15 and race-lap clip. `tracks.py` passes it; `TrackOut.typical_stint_laps: Optional[Dict[str, Optional[int]]] = None` (extra/optional, backward compatible). Contract `team/contracts/track-optimizer.md` and `ASSUMPTIONS` text updated. Frontend: optional field only, nothing required (`params_used.effective_max_stint_laps` already shows the cap in use).
- Tests: `test_track_params.KEYS` gained the field; new `test_optimizer.test_typical_stint_laps_tightens_cap_and_none_is_ignored`.

| check_optimizer_plans (23 tracks) | before | after |
|---|---|---|
| top plans containing SOFT | 18 | 18 |
| top plans with 2 SOFT stints | 4 | 6 |
| 1-stop / 2-stop top plans | 19 / 4 | 16 / 7 |
| stop count matches top-10 mode (17 comparable) | 15 | 17 |
| same compound set as winner | 5 | 5 |
| median gap to real winner plan (model s) | 2.3 | 2.3 |
| max gap | 24.9 | 18.9 |
| Austin top plan | H28/S28 | H28/S28 (unchanged) |

Honest read: the p90 cap did not remove SOFT from plans. Austin is unchanged because 56 laps with a 2-compound rule and 28-lap caps forces ~28/28 anyway, and SOFT p90 there equals max (28) (few stints). The cap is not what drives the SOFT preference; the fitted offsets (SOFT -0.12 s, wear equal to HARD) are. Tighter caps push some tracks to 2 stops, which agrees better with real stop counts (17/17 comparable).

## 2. Recalibration on 16 races
| constant | old | new | change | action |
|---|---|---|---|---|
| DEFAULT_LAP_TIME_NOISE_STD | 0.8 | calibrate_lap_noise.py: std 0.992 (14,164 laps, 16 races) | +24% | left |
| DEFAULT_PIT_LOSS_SECONDS | 23.5 | median 22.88 (294 clean stops, 16 races, excl. Monaco) | -2.6% | left |
| DEFAULT_PIT_LOSS_NOISE_STD | 5.0 | 5.06 (same sample) | +1% | left |

- Lap noise: moved >10%, but the script prints no recommendation, and the 0.8 was a MAD-robust figure whereas the script prints a raw std; T-002 already noted the script does not reproduce 0.8 (it gave 1.03 on 10 races, so 0.99 is if anything slightly lower). Not apples to apples, so left per the rule. Suggest the data scientist decide whether to switch the script to MAD and then retune.
- `calibrate_pit_loss.py` is hard-wired to the original 10 FastF1 sessions (still prints 369 stops, median 23.80, std 7.67 incl. Monaco); it cannot see the 16 races. The 16-race figures above came from an ad hoc computation over `data/processed/*_pit_stops.csv` using the same filter as `build_track_params.pit_durations`. A proper fix is the data scientist's.
- Comments in `monte_carlo.py` / `race_simulator.py` still say "10 races"; not edited (values unchanged).
