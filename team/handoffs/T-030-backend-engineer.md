# T-030 — backend-engineer — use_track_wear flag + safety cars / VSC

**Status:** done. pytest 326 passed (+6 new in `test_t030_backend.py`; one legit pin changed: `test_t007_qa` top-level /simulate keys now include the two optional new fields). `validate_replay.py` unchanged (loop untouched). **Next:** qa-engineer; frontend if it wants toggles.

## 1. use_track_wear (POST /optimize)
- Request: `use_track_wear: bool = false`. When true and the track has a T-028 fit, `track_id` is passed to `optimize` (per-track wear). Falls back to global silently if no fit.
- Response: `params_used.use_track_wear: Optional[bool]` = what was actually applied; an extra assumptions line is appended when applied. Default behaviour byte-identical.

## 2. Safety cars / VSC (opt-in, default false)
New request fields (exact): `POST /optimize` `safety_cars: bool = false`; `POST /simulate` `safety_cars: bool = false`.
New optional response fields: `/optimize` `params_used.safety_cars: Optional[bool]`; `/simulate` top-level `safety_cars: Optional[bool]` and `assumptions: Optional[List[str]]`. On /optimize with the flag on, the "No safety cars..." assumption is replaced by an "Safety cars ON: ..." line. The /simulate `lap_times` trace stays the green run.

Code: `core/safety_cars.py` (params loader, per-iteration SC/VSC timeline, per-strategy extra seconds), `simulate_totals/compare_strategies/optimize(..., safety_cars=None)`. Timeline drawn after the noise draws from a child RNG, so safety_cars off is bit-identical to before, and on/off runs share the same noise. One timeline per iteration shared by all strategies (CRN). Fitted by new `calibrate_safety_cars.py` -> `data/tracks/safety_car_params.json` (40 races: 16 in data/processed + usable 2025/2026 staging; lap is SC if >50% of drivers flagged).

**Fitted numbers**
| | SC | VSC |
|---|---|---|
| P(>=1 per race) | 0.525 (21/40) | 0.50 |
| events per race | 0.65 (26 events) | 0.75 (30 events) |
| length laps median / mean | 5 / 5.6 (range 1-13) | 2 / 2.3 |
| starts on lap 1 | 19% | 0% |
| start position | front-loaded (7 of 26 in first tenth), otherwise lumpy; model blends 50% flat | roughly flat |
| lap time vs green | x1.46 | x1.19 |
| pit loss (in+out lap excess vs field) vs green | 0.88 (n=190 vs 806 green) | 0.68 (n=47) |

Honest caveats: the pit-loss ratio is the weak one. 0.88 for SC is far above the usual "about half" folklore and VSC (0.68) cheaper than SC is odd, so it is probably noisy/biased by SC deployment mid-lap and by measuring from lap times. I also ran a sensitivity at 0.5 (below). Event counts are small (26/30), per-track rates are not estimated, plans are fixed (no reactive stop-under-SC, which is where the real value is), tyre wear and fuel are not reduced under SC, plans are screened without SCs and only re-ranked with them.

**Effect on 5 tracks** (optimizer, 4000 sims, seed 0; "gap" = best 2-stop minus best 1-stop mean, seconds; Monza = `italy`, Silverstone = `britain`)
| track | top plan off / on | gap off | gap on | gap on, pit ratio 0.5 | mean shift |
|---|---|---|---|---|---|
| Austin | H28/S28 / same | +13.75 | +13.23 | +12.75 | +174 s |
| Monaco | S32/H46 / same | +3.76 | +2.95 | +2.63 | +188 s |
| Monza | H29/S24 / same | +15.17 | +14.61 | +14.15 | +135 s |
| Singapore | H30/S32 / same | +11.68 | +11.12 | +10.68 | +176 s |
| Silverstone | H27/M25 / same | +18.31 | +17.59 | +17.04 | +144 s |

Read: the effect on strategy choice is small. SC laps slow every plan ~equally (+135-190 s on the mean), the top plan never changes, and 2-stops gain only 0.5-0.8 s on 1-stops (up to 1.3 s with a 0.5 pit ratio), well inside model error. Monaco's 1- vs 2-stop gap is only 3-4 s either way. To make SCs matter, the engine needs reactive pitting (stop when the SC appears) and probably SC-dependent tyre wear; both are a bigger step than this task.
