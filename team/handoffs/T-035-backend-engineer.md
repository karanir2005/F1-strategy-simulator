# T-035 backend-engineer handoff: qualifying prediction

## What changed
- `predictor.simulate()` (prequali mode) now also returns p_pole, p_front_row, p_q3, p_q1_exit (+ existing exp_grid); `predict_race` puts them in the table and the CLI CSV. `q1_cutoff(n)`: 22 cars -> slot >= 17 out (16 advance); <=20 cars -> slot >= 16. Race probabilities are bit-identical (same RNG order; the R18 pre-quali run still gives ANT 26.3% win).
- Eval: `predictor.quali_walk_forward/quali_score`, `python predict_race.py --evaluate-quali --offline` -> `data/jolpica/eval/quali_walk_forward.csv`. Target is the Jolpica qualifying classification; every prediction uses only earlier rounds (baseline rank tables use 2022-25 plus earlier rounds, as RankBaseline does).
- API: `qualifying` block, exact shape in `team/contracts/prediction-api.md` (Addendum T-035). `routes/predict.py`, `services/prediction_service.py`.
- Bug fixed: T-029 `p_win_prequali`/`p_podium_prequali` were stripped by the response model over HTTP (tests only called `shape()`). Now declared + `response_model_exclude_unset`.
- Generated `data/jolpica/predictions/2026_R18_prequali_<now>.csv` with the new columns (offline), so grid mode after qualifying has a snapshot. The two older R18 files lack the columns and are skipped. Rerun the CLI pre-quali before qualifying if you want a fresher one.

## Validation (walk-forward, mean per-race binary log-loss, lower better; Spearman higher better)
| season | method | pole LL | Q3 LL | pole NLL | Spearman | races |
|---|---|---|---|---|---|---|
| 2026 R2-17 | model | 0.1312 | 0.2993 | 1.954 | 0.849 | 16 |
| | previous-race quali order | 0.1545 | 0.4159 | 2.436 | 0.789 | |
| | championship order | 0.1371 | 0.3637 | 2.058 | 0.827 | |
| | uniform | 0.1858 | 0.6893 | 3.085 | n/a | |
| 2025 R2-24 | model | 0.1453 | 0.5071 | 1.946 | 0.686 | 23 |
| | previous-race quali order | 0.1509 | 0.5349 | 2.081 | 0.600 | |
| | championship order | 0.1662 | 0.5626 | 2.360 | 0.584 | |
| | uniform | 0.1988 | 0.6931 | 2.994 | n/a | |

The model beats every baseline on every metric in both seasons. Honest caveats: 2026 pole margin over championship order is small (0.006 LL, 16 races, within noise); the Q3 and Spearman gaps are clearer. 2025 Q3 LL 0.507 vs 0.30 in 2026 shows 2025 was a much closer field. Baselines are per-rank frequency tables normalised per race, not tuned further. Pole NLL ~1.95 means the actual pole sitter averaged ~14% probability: pole is inherently hard to call.

## Tests
`backend/tests/test_t035_quali.py` (10 tests). Full pytest: 338 passed, 1 failed: `test_api_predict.py::test_cache_expires`, which fails identically on the untouched baseline (verified via git stash); it is date-dependent (quali_date 2026-10-24 vs the UTC date triggers the 120 s quali-day TTL), not related to this change.

## Notes for others
- Frontend: build against the addendum; treat `qualifying: null` as hidden. `evidence` may be null.
- Not touched: frontend/, cache/, strategy simulator, shipped race defaults. Incident: I briefly ran `git stash` to test the baseline, which failed to restore `cache/fastf1_http_cache.sqlite` (locked); I restored my source files from the stash and left the sqlite file as it was on disk. Orchestrator may want to check `git status` for cache/.
- Next: frontend-engineer for the qualifying UI.
