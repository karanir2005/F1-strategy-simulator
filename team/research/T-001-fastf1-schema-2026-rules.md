# T-001: FastF1 schema and 2026 rules brief

Researcher, 2026-10-19. FastF1 3.8.1 (`.venv`). All "verified" claims come from loading the 10 cached 2026 races read-only with `Cache.offline_mode(True)`: Australia, China, Japan, Miami, Canada, Monaco, Austria, Britain, Belgium, Hungary. The inspection scripts live in the session scratchpad, not the repo.

**Era labels:** **[2026]** means checked against 2026 data or 2026 regulations. **[pre-2026]** means an older-era number, usable only as a prior. **[unverified]** means I could not confirm it.

---

## 0. TL;DR for T-002

1. **Use FastF1 `Stint` for stint boundaries. It is 100% reliable on this data.** In all 10 races, every `Stint` increment has `PitInTime` on the previous lap, and every pit-in is followed by a `Stint` increment (387/387, 0 exceptions). Within a stint, `TyreLife` always steps by +1 and the compound never changes. The only exception is a data gap (Hungary PER, see §2.3).
2. **Not every `Stint` increment is a tyre change.** 44 of 387 increments keep the same tyre: same compound, and `TyreLife` = previous + 1. These are pit-lane passes, not stops: Monaco "ALL CARS THROUGH THE PIT LANE" under SC (laps 61 and 66), drive-through penalties, and garage visits (ALO Australia, 972 s). Split these out, or replays will add phantom stops.
3. **Real used-set stints: 57 of the 343 actual tyre changes (17%).** Start ages are 2–9, mostly 4–5 (sets scrubbed in qualifying). The roadmap's "29% / 104" figure lumps together real used sets (57) and same-tyre pit passes (44). For T-004, `start_age` = `TyreLife` on the first lap of the stint.
4. **Gaps:** `Time` (end-of-lap session time) is 100% populated, even on laps where `LapTime` is NaN. The gap to the car ahead is a same-lap `Time` difference. I validated the gap to leader on the final lap against `session.results`: median error 0.00–0.09 s per race. The only large errors (5/10/20 s) are post-race time penalties that `results` includes and lap timing doesn't.
5. **SC/VSC/red:** decode `TrackStatus` per character: `2` yellow, `4` SC, `5` red, `6` VSC deployed, `7` VSC ending. Today's `is_green_flag = (TrackStatus == "1")` also throws away local-yellow laps (`"12"`, `"21"`). Keep it for backward compatibility, but add separate flags.
6. **2026 regulations:** DRS is gone. The `DRS` channel in `car_data` is 0 for all 33,508 samples (Britain, LEC). Overtake Mode activates when a car is within the "Detection Gap" of *any* car ahead at the Detection Line (reportedly 1 s). Race control messages log `OVERTAKE ENABLED/DISABLED` by lap. The two-compound rule still applies. The Monaco 2-stop rule was dropped for 2026. Pit-lane limit defaults to 80 km/h. Tyre changes during a red flag are free.

---

## 1. Columns that exist and are populated [2026, verified]

### 1.1 `session.laps`

Columns: `Time, Driver, DriverNumber, LapTime, LapNumber, Stint, PitOutTime, PitInTime, Sector1-3Time, Sector1-3SessionTime, SpeedI1, SpeedI2, SpeedFL, SpeedST, IsPersonalBest, Compound, TyreLife, FreshTyre, Team, LapStartTime, LapStartDate, TrackStatus, Position, Deleted, DeletedReason, FastF1Generated, IsAccurate`

| Column | Populated (10 races) | Notes / quirks found |
|---|---|---|
| `Driver` | 100% | **3-letter abbreviation** (e.g. `NOR`). The `data_loader.py` docstring says "Driver number", which is wrong. `DriverNumber` holds the number. |
| `Time` | 100% | Session time at the **end** of the lap. This is the key to gaps. |
| `LapStartTime` | 100% | `Time − LapStartTime − LapTime` is ≤0.01 s on every lap except **lap 1** (≈20 drivers per race, up to 0.42 s). Use `LapTime` for lap 1 and don't rebuild it from `Time`. |
| `LapStartDate` | **0%** (all NaT) | Don't use it. |
| `LapTime` | 97–99.5% | NaN on 1–62 laps per race, concentrated on **SC laps** (code `4`), red flag, some pit in-laps, and DNS lap-1 rows. Belgium has 62 NaN laps (SC laps 2–4). **The current loader drops these rows, which breaks position/gap continuity** (see §2.6). |
| `Sector1Time` | ~23–25 NaN per race | Lap 1 has no S1. S2 and S3 are ~99.8% populated. |
| `Stint` | 100% | Reliable, see §2.1. |
| `Compound` | 100% except Hungary PER laps 22–46 (`None`, 25 laps) | Values: `SOFT/MEDIUM/HARD/INTERMEDIATE` (14 INTER laps in Canada). **These are labels relative to the event, not compounds.** See the C-number table in §3.2. |
| `TyreLife` | 100% except the same 25 Hungary laps | Includes laps from earlier sessions on used sets (FastF1 docs). Equals 1 on the first lap of a new set. |
| `FreshTyre` | 100% | Agrees with `TyreLife==1` except in 5 cases (Canada BOR L19, HUL L21, plus 3 in Monaco) where `FreshTyre=False` and `TyreLife=1`. `FreshTyre=True` with `TyreLife>1` never happens. **Use `TyreLife` as the source of truth.** Keep `FreshTyre` only as an informational column. |
| `Position` | 99.6–99.9% (1–5 NaN per race) | Equals the rank of `Time` among cars on the same lap 97–100% of the time. The mismatches are ±1 and sit on lap 1 and SC/incident laps. |
| `PitInTime` / `PitOutTime` | ~5% each (only pit laps) | PitIn is on the in-lap, PitOut on the next lap. A few lap-1 `PitOutTime` values come from pit-lane starters (Miami, Canada, Monaco, Britain, Hungary: 1 each). |
| `TrackStatus` | 100% | A string of every status code seen during the lap (e.g. `"671"`). Codes seen: `1, 2, 4, 5, 6, 7`. `3` never appears. |
| `IsAccurate` | 108–275 False per race | False on in/out laps, SC/VSC laps, the first lap after SC, laps missing sector times, and lap 1 ([FastF1 docs](https://docs.fastf1.dev/core.html)). A good extra filter for fitting. |
| `Deleted` | 5–35 per race | Track-limits deletions. The lap was still driven, so it's fine for race pace. Keep the column and don't filter on it. |
| `FastF1Generated` | 1–5 per race | Synthetic rows (e.g. partial last lap of a retiree). **Drop them for fitting.** |
| `Team` | 100% | Needed for team-mean pace shrinkage in Phase 1. |

### 1.2 `session.results` [2026, verified]

Columns: `DriverNumber, BroadcastName, Abbreviation, DriverId, TeamName, TeamColor, TeamId, FirstName, LastName, FullName, HeadshotUrl, CountryCode, Position, ClassifiedPosition, GridPosition, Q1, Q2, Q3, Time, Status, Points, Laps`

- `Status` values in 2026 data are only `Finished`, `Lapped`, `Retired`, and `Did not start`. There are no DNF causes such as "Gearbox".
- **`Time`:** the winner's total race time. For other lead-lap finishers it's the **gap to the winner, including post-race time penalties**. This was observed in the data and contradicts the docs' "total race time". For `Lapped` cars it isn't a usable gap (e.g. SAI Britain: 4.39 s with 51 laps). Use `Laps` + `Position` for those.
- `ClassifiedPosition`: `"R"` = retired and not classified. `GridPosition` is never 0/NaN in these 10 races. In FastF1 convention, 0 = pit-lane start [unverified on 2026].
- **DNS drivers can still have a laps row.** China PIA/NOR/BOR/ALB, Australia HUL and Canada LIN each have one lap-1 row with NaN `LapTime`. Drop drivers whose `Status == "Did not start"`.

### 1.3 `session.weather_data` [2026, verified]

`Time, AirTemp, Humidity, Pressure, Rainfall, TrackTemp, WindDirection, WindSpeed`. There are ~146–208 rows per race (about one per minute), keyed on session `Time`, so you can `merge_asof` them onto laps by `Time`. Rainfall was True in Miami only, for 3 rows near the end of the session. Nobody ran INTERs there, so it doesn't count as a "wet race".

### 1.4 Other useful tables [2026, verified]

- `session.track_status`: `Time, Status, Message` intervals (`AllClear, Yellow, SCDeployed, VSCDeployed, VSCEnding, Red`). Gives SC/VSC durations to the second, which is better than per-lap flags for Phase 2 SC statistics.
- `session.race_control_messages`: `Time` (wall clock, **not** session time), `Category, Message, Status, Flag, Scope, Sector, RacingNumber, Lap`. Useful messages: `SAFETY CAR DEPLOYED`, `SAFETY CAR IN THIS LAP`, `VSC DEPLOYED/ENDING`, `RED FLAG - RACE SUSPENDED`, **`OVERTAKE ENABLED/DISABLED`**, `ALL CARS THROUGH THE PIT LANE`, `PIT LANE ENTRY CLOSED/OPEN`, and penalty texts. Beware: a naive regex for "RED FLAG" also matches "CHEQUE**RED FLAG**".
- `session.total_laps`: scheduled race distance.

---

## 2. Derivation recipes for T-002

All recipes assume `L = session.laps.sort_values(["Driver","LapNumber"])` and `g = L.groupby("Driver")`.

### 2.1 Stint id and tyre-change detection [2026, verified]

```python
prev_stint    = g["Stint"].shift(1)
new_stint     = prev_stint.notna() & (L["Stint"] != prev_stint)   # == pit-in on previous lap, 387/387
same_tyre     = new_stint & (L["Compound"] == g["Compound"].shift(1)) \
                          & (L["TyreLife"] == g["TyreLife"].shift(1) + 1)
tyre_change   = new_stint & ~same_tyre          # 343 real changes in 10 races
# tyre stint id: increments only on a real tyre change (lap 1 = stint 1)
L["tyre_stint"]  = (tyre_change.astype(int)).groupby(L["Driver"]).cumsum() + 1
L["stint_id"]    = L["Driver"] + "_" + L["tyre_stint"].astype(str)
L["stint_start_age"] = L.groupby(["Driver","tyre_stint"])["TyreLife"].transform("first")  # -> T-004 start_age
L["is_pit_pass_no_change"] = same_tyre          # drive-through / SC pit-lane pass / garage
```

- Keep FastF1's raw `Stint` as `ff1_stint` as well, for traceability.
- In `clean_laps`, drop **both** the in-lap (`PitInTime` set) and the out-lap (`PitOutTime` set, i.e. the first lap of any `new_stint`) whatever the tyre age. The old logic only caught out-laps on `TyreLife==1`.
- Counts per race (stint increments / same-tyre / real used-set starts): AUS 32/3/0, CHN 19/0/0, JPN 29/3/0, MIA 22/1/1, CAN 34/1/6, MON 86/35/21, AUT 41/0/2, GBR 51/1/16, BEL 28/0/3, HUN 45/0/8.
- Monaco is pathological: an SC led the field through the pit lane twice, followed by a red flag with free tyre changes. 16 tyre changes in the dataset touch red-flag laps. Flag them (§2.4) so that pit-loss calibration and replays can exclude or special-case them.
- Lap-1 used sets also occur (start on a used set: China 2, Canada 7, Hungary 1). `stint_start_age` handles these.

### 2.2 Gap to car ahead [2026, verified]

Two definitions. Extract both, because they differ on 4–43% of laps (lapped traffic):

**(a) Race-order gap (same lap, by `Position`)** for undercut/overcut and position battles:
```python
G = L[["Driver","LapNumber","Position","Time"]].dropna(subset=["Position"])
ahead = G.assign(Position=G.Position + 1).rename(columns={"Time":"time_ahead","Driver":"car_ahead"})
G = G.merge(ahead[["LapNumber","Position","time_ahead","car_ahead"]], on=["LapNumber","Position"], how="left")
G["gap_ahead_s"]  = (G.Time - G.time_ahead).dt.total_seconds()      # NaN for P1
G["gap_leader_s"] = (G.Time - G.groupby("LapNumber").Time.transform("min")).dt.total_seconds()
```
Check: 0 negative gaps and 0 missing gaps (P>1) across all 10 races. 14–37% of laps are within 1.0 s of the car ahead (median gap 1.5–6.5 s depending on the race).

**(b) On-road gap (any car, any lap)** for dirty air, traffic and Overtake Mode. **The Overtake Detection Gap counts any F1 car ahead, lapped cars included** (FIA B7.2.4c "another F1 Car"):
```python
X = L[["Driver","LapNumber","Time"]].dropna().sort_values("Time")
X["road_car_ahead"]  = X.Driver.shift(1)
X["road_gap_ahead_s"] = X.Time.diff().dt.total_seconds()
X["road_ahead_lap_diff"] = X.LapNumber.shift(1) - X.LapNumber   # 0 = same lap; ≠0 = lapping/lapped traffic
```
Caveats: both are measured **at the timing line**, not at each circuit's Detection Line. That's fine as a proxy. Gaps on laps containing `4`/`5`/`6`/`7` aren't racing gaps, so mask them via the flags in §2.3. You can also compute gaps per sector by using `SectorNSessionTime` instead of `Time`.

### 2.3 Track-status flags [2026, verified; codes from [FastF1 api docs](https://docs.fastf1.dev/api.html)]

```python
ts = L["TrackStatus"].fillna("")
L["is_green_flag"] = ts == "1"                      # keep (backward compat; current tire-model filter)
L["has_yellow"]    = ts.str.contains("2")
L["has_sc"]        = ts.str.contains("4")
L["has_vsc"]       = ts.str.contains("[67]")         # 6 = VSC deployed, 7 = VSC ending
L["has_red"]       = ts.str.contains("5")
L["is_neutralised"] = L.has_sc | L.has_vsc | L.has_red
L["is_racing_lap"] = ~L.is_neutralised               # yellow-only laps count as racing (recommend for fitting)
```

- Number of laps containing each code, across the 10 races: `1` 10,917, `2` 716, `4` 735, `6` 445, `7` 273, `5` 16.
- Lap-time ratio vs the green median [2026, data]: pure SC laps 1.31–1.52×, pure VSC laps 1.34–1.37×. Use these as Phase 2 priors.
- SC/VSC events in the sample (from `track_status`): **SC 8, VSC 13, red 1 in 10 races.** Per race: AUS 3 VSC; CHN 1 SC; JPN 1 SC; MIA 1 SC; CAN 3 VSC; MON 3 SC + red; AUT 2 VSC; GBR 2 VSC + 1 SC; BEL 1 SC (lap 1) + 2 VSC; HUN 1 VSC.
- Pit stops whose in-lap was under SC/VSC/red are a large share: AUS 22/32, CHN 9/19, JPN 14/29, MON 64/86, BEL 17/28, GBR 23/51. Add `pit_under_sc` / `pit_under_vsc` / `pit_under_red` (from the in-lap's flags) so pit-loss calibration can separate cheap stops from green-flag stops.
- Hungary PER laps 22–46 have `Compound=None` and `TyreLife=NaN` inside a valid stint 2. Forward-fill `Compound` within `Stint` and rebuild `TyreLife = first_age + (LapNumber − first_lap)`. If the first age is missing, mark the stint `tyre_data_imputed=True` or drop it from fitting.

### 2.4 Pit-lane time [2026, verified]

`pit_lane_s = PitOutTime(out-lap) − PitInTime(previous lap)`, which is what `extract_pit_stops` already does. Medians per race for real tyre changes, in seconds: AUS 19.1 (VSC-heavy), CHN 23.4, JPN 24.2, MIA 23.2, CAN 25.3, MON 31.5 (red-flag contaminated; p90 = 2151 s), AUT 21.6, GBR 29.5, BEL 24.7, HUN 22.0. This is pit-lane **traversal** time, not net race-time loss, and it is clearly **track-dependent** (Phase 3 per-track pit loss). Exclude red-flag stops and `is_pit_pass_no_change` rows before recalibrating.

### 2.5 Overtake Mode state per lap [2026, verified that the messages exist]

Race-control messages carry `OVERTAKE DISABLED` / `OVERTAKE ENABLED` with `Lap`. All 10 races show `DISABLED` then `ENABLED` on lap 1, and re-enable messages after each SC (e.g. CHN L14, JPN L28, MIA L12, MON L71). Recommended derivation: `overtake_enabled = False` on lap 1 and on any lap between a `DISABLED` message and the next `ENABLED` message. The approximation `~has_sc & LapNumber>1` agrees with the regulation (§3.1). Per-car activation (within the Detection Gap) is **not** in FastF1. Infer it from `road_gap_ahead_s < 1.0`.

### 2.6 Row retention

The current `extract_laps_df` drops every row with NaN `LapTime`. Recommendations:
- Keep all rows except `FastF1Generated` and DNS drivers.
- Expose `lap_time_seconds` as nullable.
- Let `clean_laps` (fit filter) drop NaN lap times.

Gaps and positions need the full sequence, and `Time` is 100% populated even when `LapTime` isn't.

---

## 3. 2026 regulations relevant to the sim

Primary source: **FIA 2026 F1 Regulations, Section B Sporting, Issue 07 (25 Jun 2026)**, [PDF](https://api.fia.com/system/files/documents/fia_2026_f1_regulations_-_section_b_sporting_-_iss_07_-_2026-06-25.pdf). Text was extracted with `pdftotext`. Some sub-article numbers came out garbled in the layout extraction. Article headings are as cited.

### 3.1 Overtake Mode replaces DRS [2026]

| Fact | Source |
|---|---|
| No DRS. "Driver Adjustable Bodywork" (front wing + rear flap) switches between Corner Mode and Straight Mode in FIA-defined Activation Zones. It is **available to every car, regardless of gap**. Only partial activation is allowed in low grip. | FIA B7.1 |
| **Overtake** is an extra ERS-K power curve. It is activated at the Activation Line if the car was **less than the "Detection Gap" behind another F1 car** (any car, lapped cars included) at the Detection Line. The gap value and line positions are set per circuit and published ≥4 weeks before the event. | FIA B7.2.1b(vii–ix), B7.2.4c |
| Detection Gap is reportedly **1.0 s**, carried over from DRS. The regulation itself doesn't fix the number. **[unverified per circuit]** | [Sky Sports](https://www.skysports.com/f1/news/12040/13484743/f1-2026-f1-announces-new-racing-rules-for-drs-replacement-with-arrival-of-overtake-mode-plus-boost-mode-and-active-aero), [ESPN](https://www.espn.com/f1/story/_/id/47333501/); [PlanetF1](https://www.planetf1.com/news/explained-how-f1-new-overtaking-mode-will-work) says "not set in stone" |
| Overtake is **disabled at the race start and at every restart** until the leader crosses the Detection Line. It is **disabled under SC** and re-enabled once all cars have crossed the Line after the SC enters the pit entry. The Race Director can disable it at will, and does so in low grip. | FIA B7.2.2 "Enabling & Disabling of Overtake". Confirmed by RCM in all 10 races (§2.5) |
| Magnitude: standard-mode ERS-K tapers to 0 at ~345 km/h, Overtake mode at ~355 km/h. That's ~110 kW vs ~340 kW at 338 km/h, plus +0.5 MJ allowable recharge on a lap with Overtake (8.5 → 9.0 MJ). **[secondary source; C5.2.8/C5.2.10 not checked]** | [PlanetF1 explainer](https://www.planetf1.com/news/explained-how-f1-new-overtaking-mode-will-work) |
| `car_data["DRS"]` is 0 for all 33,508 samples (Britain 2026, LEC). Overtake/active-aero state is **not** exposed in FastF1 telemetry. | Verified in cache |

**Modelling implication:** pre-2026 DRS overtake probabilities and DRS-train behaviour are priors only **[pre-2026]**. Fit the Phase 1 overtake probability from 2026 position changes vs `road_gap_ahead_s`, pace delta and tyre-age delta. Use 1.0 s as the natural feature breakpoint, and exclude lap 1 and SC restart laps (Overtake disabled).

### 3.2 Tyres [2026]

| Fact | Source |
|---|---|
| Five dry compounds, **C1–C5** (C6 dropped for 2026). Three are nominated per event as Hard/Medium/Soft. | [Pirelli press](https://press.pirelli.com/the-range-of-compounds-for-the-2026-season-has-been-set/), FIA B6.1.1a |
| Tyres are narrower (front −25 mm, rear −30 mm). **[plan as of Jun 2025]** | [The Race](https://www.the-race.com/formula-1/two-big-tyre-changes-for-f1-2026-revealed/) |
| Max sets per driver per event: **dry 13 (standard format) / 12 (sprint format); intermediate 5; wet 2.** Q3 cars get an extra soft set, which must be returned. Two sets of the mandatory race spec must be kept for the race. Pre-2026 commonly cited 4 inters / 3 wets **[pre-2026, unverified]**. | FIA B6.3.4 table, B6.3.9a |
| **Two-compound rule:** unless INT/WET were used, each driver must use ≥2 different dry specifications in the race, at least one of which is a nominated mandatory race spec (up to 2 nominated, announced 2 weeks before). Breach = DSQ. If the race is suspended and not resumed, +30 s instead. | FIA B6.3.8, B6.1.2b(ii) |
| **Monaco 2-stop / 3-set rule (2025 only) is dropped for 2026.** | [PlanetF1, 28 Feb 2026](https://www.planetf1.com/news/monaco-grand-prix-two-stop-rule-scrapped-2026-fia); Mercedes Monaco preview |
| Sprint-format events among the pulled races: **China, Miami, Canada, Britain** (dry allocation e.g. 6S/4M/2H). | FastF1 `get_event_schedule(2026).EventFormat` (verified) + Mercedes previews |

**Compound mapping for the pulled races [2026].** FastF1 only reports S/M/H. Add a `compound_c` column from this table, since "SOFT" is C3 at Suzuka and C5 at Monaco:

| Race | Hard | Medium | Soft | Source |
|---|---|---|---|---|
| Australia | C3 | C4 | C5 | [Mercedes](https://www.mercedesamgf1.com/news/what-are-the-tyre-options-for-the-2026-australian-gp) (via search snippet) |
| China | C2 | C3 | C4 | [Mercedes](https://www.mercedesamgf1.com/news/what-are-the-tyre-options-for-the-2026-chinese-gp) |
| Japan | C1 | C2 | C3 | [Mercedes](https://www.mercedesamgf1.com/news/what-are-the-tyre-options-for-the-2026-japanese-gp) (via search snippet) |
| Miami | C3 | C4 | C5 | [Coffee Corner Motorsport](https://coffeecornermotorsport.com/f1-miami-grand-prix-2026-tyre-preview/) **[secondary]** |
| Canada | C3 | C4 | C5 | [Mercedes](https://www.mercedesamgf1.com/news/what-are-the-tyre-options-for-the-2026-canadian-gp) |
| Monaco | C3 | C4 | C5 | [Mercedes](https://www.mercedesamgf1.com/news/what-are-the-tyre-options-for-the-2026-monaco-gp) |
| Austria | C3 | C4 | C5 | [Mercedes](https://www.mercedesamgf1.com/news/what-are-the-tyre-options-for-the-2026-austrian-gp) |
| Britain | C1 | C2 | C3 | [Mercedes](https://www.mercedesamgf1.com/news/what-are-the-tyre-options-for-the-2026-british-gp) |
| Belgium | C2 | C3 | C4 | [Mercedes](https://www.mercedesamgf1.com/news/what-are-the-tyre-options-for-the-2026-belgian-gp) |
| Hungary | C3 | C4 | C5 | [Mercedes](https://www.mercedesamgf1.com/news/what-are-the-tyre-options-for-the-2026-hungarian-gp) |

**Modelling implication:** pooling "SOFT" across races mixes C3 with C5. That is a likely contributor to the inverted SOFT pace offset noted in the roadmap. The tire model should key on `compound_c` (with partial pooling by track) or include a race effect.

### 3.3 Pit lane, SC, VSC, red flag [2026]

| Fact | Source |
|---|---|
| Pit-lane speed limit is **80 km/h for the whole event**, but the Race Director may amend it. Monaco reportedly 60 km/h **[unverified]**. Australia was planned at 60 but stayed at 80. | FIA B1.6 "Driving in the Pit Entry Road, Pit Lane and Pit Exit Road" (a); [Speedcafe](https://us.speedcafe.com/f1-news-2026-fia-clarify-pit-lane-rule-speed-limit-albert-park-australian-grand-prix/) |
| Pit-loss implication: no 2026 rule change to the limit itself. Use the measured per-track pit-lane times (§2.4, 19–31 s) rather than a 2025 prior. A 60 vs 80 km/h difference is ~5 s per stop **[pre-2026 estimate, secondary]**. | §2.4 data; [RacingNews365](https://racingnews365.com/fia-set-to-raise-f1-pit-lane-speed-limit-report) |
| **VSC:** cars may enter the pits under VSC **only to change tyres**. Must stay above the FIA minimum (delta) time. Every VSC lap counts as a race lap. | FIA B5.12 "Use of Pit Lane during VSC Deployment" |
| **SC:** lapped cars may unlap (`LAPPED CARS MAY NOW OVERTAKE`). The SC can route the field through the pit lane (`ALL CARS THROUGH THE PIT LANE`, seen at Monaco 2026). | FIA B5.13; RCM verified |
| **Red flag:** cars stop in the fast lane, and "changing wheels and tyres" is permitted, so **a red flag is a free tyre change**. Race suspension time is added to the max race duration. | FIA B5.14.4a(vii), B5.14.2 |
| **Car:** 768 kg minimum (vs 800 kg in 2025), excluding fuel. **Race fuel load and fuel-burn: [unverified for 2026]**. The 110 kg tank cap is a [pre-2026] figure. Keep fitting the fuel coefficient from data. | [Red Bull guide](https://www.redbullracing.com/int-en/projects/bulls-guide-to-the-f1-2026-regulations/technical-regulations), [GPFans](https://www.gpfans.com/en/f1-news/1077929/f1-2026-regulations-explained/) |

---

## 4. Recommended column list for T-002

### `data/processed/{gp}_{year}_laps.csv`
Existing columns are kept so `tire_model.py` doesn't break. New columns are marked with *.

| Column | Type | Recipe |
|---|---|---|
| `driver` | str | `Driver` (abbreviation; fix the docstring) |
| `driver_name` | str | keep (currently duplicates `driver`) |
| `driver_number`* | str | `DriverNumber` |
| `team`* | str | `Team` |
| `lap_number` | int | `LapNumber` |
| `lap_time_seconds` | float, **nullable** | `LapTime` (s) |
| `lap_start_s`*, `lap_end_s`* | float | `LapStartTime`, `Time` (session seconds) |
| `s1_s`*, `s2_s`*, `s3_s`* | float | `Sector{n}Time` |
| `compound` | str | `Compound`, ffilled within `Stint` |
| `compound_c`* | str | §3.2 mapping (`C1`..`C5`; `INTER`/`WET` passthrough) |
| `tire_age` | float | `TyreLife` (rebuilt if NaN, §2.3) |
| `fresh_tyre`* | bool | `FreshTyre` (informational) |
| `ff1_stint`* | int | raw `Stint` |
| `tyre_stint`*, `stint_id`* | int, str | §2.1 |
| `stint_start_age`* | int | §2.1 → T-004 `start_age` |
| `pit_in_lap` | int | keep (`PitInTime` notna) |
| `pit_out_lap`* | int | `PitOutTime` notna |
| `pit_lane_s`* | float | §2.4, on the out-lap |
| `is_pit_pass_no_change`* | bool | §2.1 |
| `pit_under_sc`*, `pit_under_vsc`*, `pit_under_red`* | bool | flags of the in-lap |
| `position`* | float, nullable | `Position` |
| `gap_ahead_s`*, `car_ahead`*, `gap_leader_s`* | float/str | §2.2a |
| `road_gap_ahead_s`*, `road_car_ahead`*, `road_ahead_lap_diff`* | float/str/int | §2.2b |
| `track_status`* | str | raw `TrackStatus` |
| `is_green_flag` | bool | keep (`== "1"`) |
| `has_yellow`*, `has_sc`*, `has_vsc`*, `has_red`*, `is_neutralised`* | bool | §2.3 |
| `overtake_enabled`* | bool | §2.5 |
| `is_accurate`*, `deleted`*, `ff1_generated`* | bool | `IsAccurate`, `Deleted`, `FastF1Generated` |
| `track_temp_c`*, `air_temp_c`*, `rainfall`* | float/bool | `merge_asof(weather_data, on Time)` at lap end |

### Sidecars (new)
- `{gp}_{year}_results.csv`: `driver, driver_number, team, grid_position, position, classified_position, status, laps, points, time_s` (winner = total, others = gap incl. penalties, `Lapped` → NaN), `q1_s, q2_s, q3_s`.
- `{gp}_{year}_weather.csv`: raw `weather_data` with `Time` → `time_s`.
- `{gp}_{year}_track_status.csv`: `track_status` intervals (`time_s, status, message`). This is the Phase 2 SC/VSC duration source.
- `{gp}_{year}_rcm.csv`: race control messages filtered to `Category in {SafetyCar, Flag}` plus messages matching `OVERTAKE (EN|DIS)ABLED|THROUGH THE PIT LANE|PIT LANE ENTRY|RED FLAG - |PENALTY`, keeping `Lap`.
- `race_info` additions: `total_laps` (`session.total_laps`), `event_format` (`conventional` / `sprint_qualifying`), `event_date`, `compound_map`.

---

## 5. Other findings and risks

- **`pull_races.py`: `(2026, "Spain")` resolves to the Madrid "Spanish Grand Prix" (2026-09-13), not Barcelona.** Barcelona is "Barcelona Grand Prix", 2026-06-14. The script comment says Barcelona. The Spain cache folder is empty, so this race was never pulled. The T-002 re-pull needs network access and an explicit choice between `"Barcelona"` and `"Spanish"`/Madrid.
- Pre-race sessions (Q, FP2 long runs) are not cached for any race. Phase 1 pace-from-qualifying will need new pulls.
- `results.Time` includes post-race penalties, and lap timing doesn't. The finishing-order scorecard (T-003) should compare against `results.Position`, not reconstructed on-track order.
- Monaco 2026 alone contributes 35 of the 44 same-tyre pit passes and all 16 red-flag tyre changes. Consider excluding it from pit-loss calibration.
