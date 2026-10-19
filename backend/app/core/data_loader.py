"""
Data loader module for FastF1 race session data.
Extracts lap-by-lap metrics (plus results / weather / track-status /
race-control sidecars) for simulation and model fitting.

REWRITTEN (2026-10-19, T-002) per team/research/T-001-fastf1-schema-2026-rules.md:

- Stint boundaries now come from FastF1's own `Stint` column (a pit-in on
  the previous lap), split into REAL tyre changes vs same-tyre pit-lane
  passes. The old rule -- "a new stint starts when tire_age == 1" -- merged
  every stop onto a used set (57 of 343 real tyre changes in the 10 races,
  start ages 2-9) into the previous stint, and let those stints' out-laps
  (which include part of the pit lane, ~+20 s) through the tire-model fit.
- Rows with a missing LapTime are KEPT (lap_time_seconds is nullable).
  They are mostly safety-car / red-flag laps, and dropping them broke
  position/gap continuity and made a driver's "laps in stint" count wrong
  for replays. Fitting code must filter NaN lap times itself
  (tire_model.clean_laps does).
- Rows that are not real laps are DROPPED: FastF1Generated rows (synthetic
  partial last laps of retirees) and every row of a driver whose results
  Status is "Did not start" (they still get a lap-1 row with NaN LapTime).
- TrackStatus is decoded per character (2 yellow, 4 SC, 5 red, 6/7 VSC).
  is_green_flag (== "1") is kept unchanged for backward compatibility.
- compound_c records which Pirelli C-compound "SOFT/MEDIUM/HARD" meant at
  that event -- SOFT is C3 at Suzuka/Silverstone and C5 at Monaco, so
  pooling raw labels across races mixes very different tyres.
"""

import json
import os
import re
from typing import Dict, Optional, Tuple

import fastf1
import numpy as np
import pandas as pd

# Per-event Pirelli nominations, keyed by (year, FastF1 EventName) so the
# lookup doesn't depend on how the caller spelled the `gp` argument.
# Source: T-001 research note §3.2 (Mercedes / Pirelli tyre previews).
# Miami's mapping comes from a secondary source only -- treat it as less
# certain than the others. A missing event simply yields compound_c = NaN
# (with a warning), never a guessed mapping.
COMPOUND_ALLOCATION: Dict[Tuple[int, str], Dict[str, str]] = {
    (2026, "Australian Grand Prix"): {"HARD": "C3", "MEDIUM": "C4", "SOFT": "C5"},
    (2026, "Chinese Grand Prix"): {"HARD": "C2", "MEDIUM": "C3", "SOFT": "C4"},
    (2026, "Japanese Grand Prix"): {"HARD": "C1", "MEDIUM": "C2", "SOFT": "C3"},
    (2026, "Miami Grand Prix"): {"HARD": "C3", "MEDIUM": "C4", "SOFT": "C5"},
    (2026, "Canadian Grand Prix"): {"HARD": "C3", "MEDIUM": "C4", "SOFT": "C5"},
    (2026, "Monaco Grand Prix"): {"HARD": "C3", "MEDIUM": "C4", "SOFT": "C5"},
    (2026, "Austrian Grand Prix"): {"HARD": "C3", "MEDIUM": "C4", "SOFT": "C5"},
    (2026, "British Grand Prix"): {"HARD": "C1", "MEDIUM": "C2", "SOFT": "C3"},
    (2026, "Belgian Grand Prix"): {"HARD": "C2", "MEDIUM": "C3", "SOFT": "C4"},
    (2026, "Hungarian Grand Prix"): {"HARD": "C3", "MEDIUM": "C4", "SOFT": "C5"},
}
COMPOUND_ALLOCATION_SOURCE = {
    (2026, "Miami Grand Prix"): "secondary (Coffee Corner Motorsport preview)",
}
# rain tyres have no C-number; pass them through so compound_c is never
# NaN for a known compound
_WET_PASSTHROUGH = {"INTERMEDIATE": "INTER", "WET": "WET"}

# Race-control messages worth keeping in the rcm sidecar (T-001 §4).
# NOTE: a naive "RED FLAG" pattern also matches "CHEQUERED FLAG", hence
# "RED FLAG - ".
_RCM_KEEP_PATTERN = re.compile(
    r"OVERTAKE (?:EN|DIS)ABLED|THROUGH THE PIT LANE|PIT LANE ENTRY|RED FLAG - |PENALTY"
)
_OVERTAKE_PATTERN = re.compile(r"OVERTAKE\s+(ENABLED|DISABLED)")


def load_race_session(year: int, gp: str, session_type: str = "R") -> fastf1.core.Session:
    """
    Load a FastF1 race session and cache the data.

    Args:
        year: Race year (e.g., 2024)
        gp: Grand Prix name (e.g., 'Bahrain')
        session_type: 'R' for Race, 'Q' for Qualifying

    Returns:
        Loaded FastF1 Session object
    """
    session = fastf1.get_session(year, gp, session_type)
    session.load()
    return session


def _td_seconds(series: pd.Series) -> pd.Series:
    """Timedelta series -> float seconds (NaT -> NaN)."""
    if not pd.api.types.is_timedelta64_dtype(series):
        # an all-NaT column (e.g. Q1 for a race with no quali data) can
        # arrive typed as datetime64 or object
        if series.isna().all():
            return pd.Series(np.nan, index=series.index)
        series = pd.to_timedelta(series)
    return series.dt.total_seconds()


def compound_map_for_event(year: int, event_name: str) -> Optional[Dict[str, str]]:
    """Pirelli S/M/H -> C-number mapping for one event, or None if unknown."""
    return COMPOUND_ALLOCATION.get((year, event_name))


def _dns_drivers(session) -> set:
    """Abbreviations of drivers whose results Status is 'Did not start'."""
    try:
        results = session.results
    except Exception:  # results not loaded / unavailable
        return set()
    if results is None or len(results) == 0 or "Status" not in results:
        return set()
    return set(results.loc[results["Status"] == "Did not start", "Abbreviation"])


def _impute_tyre_data(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing Compound / TyreLife inside a FastF1 stint where possible.

    Within one FastF1 `Stint` the compound never changes and TyreLife steps
    by exactly +1 per lap (T-001 §2.1, verified on all 10 races), so a gap
    can be filled from any known lap of the same stint:
      Compound -> the stint's known compound (ffill + bfill)
      TyreLife -> anchor_age + (LapNumber - anchor_lap), anchored on the
                  stint's first lap that has a TyreLife.
    If the WHOLE stint has no tyre data (Hungary 2026 PER, laps 22-46 --
    the only case in the pulled races) nothing can be inferred: the
    previous stint ended in a real pit stop, so forward-filling across it
    would invent a compound. Those rows keep compound/tire_age NaN and get
    tyre_data_missing=True; tire_model.clean_laps drops them.
    """
    laps = laps.copy()
    key = [laps["Driver"], laps["Stint"]]
    orig_compound_na = laps["Compound"].isna()
    orig_age_na = laps["TyreLife"].isna()

    laps["Compound"] = laps.groupby(key, dropna=False)["Compound"].transform(lambda s: s.ffill().bfill())

    if orig_age_na.any():
        gaps = laps.loc[orig_age_na, ["Driver", "Stint"]].drop_duplicates()
        for driver, stint in gaps.itertuples(index=False):
            idx = laps.index[(laps["Driver"] == driver) & (laps["Stint"] == stint)]
            group = laps.loc[idx]
            known = group["TyreLife"].notna()
            if not known.any():
                continue  # whole stint unknown -> stays missing
            anchor = group.loc[known].iloc[0]
            rebuilt = anchor["TyreLife"] + (group["LapNumber"] - anchor["LapNumber"])
            laps.loc[idx, "TyreLife"] = group["TyreLife"].where(known, rebuilt)
    laps["tyre_data_imputed"] = (orig_compound_na & laps["Compound"].notna()) | (
        orig_age_na & laps["TyreLife"].notna()
    )
    laps["tyre_data_missing"] = laps["Compound"].isna() | laps["TyreLife"].isna()
    return laps


def _assign_tyre_stints(laps: pd.DataFrame) -> pd.DataFrame:
    """
    T-001 §2.1: a FastF1 Stint increment (== pit-in on the previous lap,
    387/387 in the 10 races) is a REAL tyre change unless the compound is
    unchanged AND TyreLife == previous lap's TyreLife + 1 -- that's a
    same-tyre pit-lane pass (Monaco's SC-led pit-lane passes, drive-
    throughs, garage visits). Defensive extra: a compound change or a
    TyreLife that doesn't continue *within* a FastF1 stint also starts a
    new tyre stint (never observed, but cheap insurance against a FastF1
    stint-numbering glitch silently merging two sets).

    Requires laps sorted by Driver, LapNumber. Adds ff1_stint, tyre_stint,
    stint_id, stint_start_age, new_ff1_stint, is_pit_pass_no_change,
    is_tyre_change.
    """
    g = laps.groupby("Driver")
    prev_stint = g["Stint"].shift(1)
    prev_compound = g["Compound"].shift(1)
    prev_age = g["TyreLife"].shift(1)
    has_prev = g.cumcount() > 0

    new_ff1_stint = has_prev & (laps["Stint"] != prev_stint)
    tyre_continues = (
        laps["Compound"].notna()
        & (laps["Compound"] == prev_compound)
        & (laps["TyreLife"] == prev_age + 1)
    )
    same_tyre_pass = new_ff1_stint & tyre_continues
    # within a FastF1 stint, only a *known* discontinuity counts -- missing
    # tyre data (NaN) must not split a stint by itself
    known_both = laps["Compound"].notna() & prev_compound.notna()
    known_ages = laps["TyreLife"].notna() & prev_age.notna()
    hidden_change = (
        has_prev
        & ~new_ff1_stint
        & (
            (known_both & (laps["Compound"] != prev_compound))
            | (known_ages & (laps["TyreLife"] != prev_age + 1))
        )
    )
    tyre_change = (new_ff1_stint & ~same_tyre_pass) | hidden_change

    laps["ff1_stint"] = laps["Stint"]
    laps["new_ff1_stint"] = new_ff1_stint
    laps["is_pit_pass_no_change"] = same_tyre_pass
    laps["is_tyre_change"] = tyre_change
    laps["tyre_stint"] = tyre_change.astype(int).groupby(laps["Driver"]).cumsum() + 1
    laps["stint_id"] = laps["Driver"] + "_" + laps["tyre_stint"].astype(str)
    laps["stint_start_age"] = laps.groupby(["Driver", "tyre_stint"])["TyreLife"].transform("first")
    return laps


def _decode_track_status(laps: pd.DataFrame) -> pd.DataFrame:
    """T-001 §2.3: per-character TrackStatus flags."""
    ts = laps["TrackStatus"].fillna("").astype(str)
    laps["track_status"] = laps["TrackStatus"]
    # kept for backward compatibility: "1" exactly == whole lap green
    laps["is_green_flag"] = ts == "1"
    laps["has_yellow"] = ts.str.contains("2", regex=False)
    laps["has_sc"] = ts.str.contains("4", regex=False)
    laps["has_vsc"] = ts.str.contains("[67]", regex=True)
    laps["has_red"] = ts.str.contains("5", regex=False)
    laps["is_neutralised"] = laps["has_sc"] | laps["has_vsc"] | laps["has_red"]
    return laps


def _add_pit_columns(laps: pd.DataFrame) -> pd.DataFrame:
    """
    pit_in_lap / pit_out_lap (0/1 ints), pit_lane_s on the out-lap
    (PitOutTime(out-lap) - PitInTime(previous lap), T-001 §2.4) and
    pit_under_sc / _vsc / _red on the IN-lap (the in-lap's track status is
    what decides whether the stop was a cheap one). Lap-1 PitOutTime
    (pit-lane starters) has no matching in-lap, so pit_lane_s stays NaN.
    """
    g = laps.groupby("Driver")
    laps["pit_in_lap"] = laps["PitInTime"].notna().astype(int)
    laps["pit_out_lap"] = laps["PitOutTime"].notna().astype(int)
    prev_pit_in = g["PitInTime"].shift(1)
    laps["pit_lane_s"] = _td_seconds(laps["PitOutTime"] - prev_pit_in)
    is_in = laps["pit_in_lap"] == 1
    laps["pit_under_sc"] = is_in & laps["has_sc"]
    laps["pit_under_vsc"] = is_in & laps["has_vsc"]
    laps["pit_under_red"] = is_in & laps["has_red"]
    return laps


def _add_gaps(laps: pd.DataFrame) -> pd.DataFrame:
    """
    T-001 §2.2, both measured at the timing line from end-of-lap `Time`:
    (a) race-order gap: same lap number, car one Position ahead
        (gap_ahead_s / car_ahead), and gap to that lap's leader;
    (b) on-road gap: whichever car crossed the line immediately before,
        on ANY lap (road_gap_ahead_s / road_car_ahead /
        road_ahead_lap_diff, 0 = same lap, >0 = that car is laps ahead).
    Gaps on neutralised laps are not racing gaps -- mask with the flags.
    """
    G = laps[["Driver", "LapNumber", "Position", "Time"]].dropna(subset=["Position", "Time"])
    ahead = G.assign(Position=G["Position"] + 1).rename(
        columns={"Time": "_time_ahead", "Driver": "car_ahead"}
    )
    merged = laps[["Driver", "LapNumber", "Position", "Time"]].merge(
        ahead[["LapNumber", "Position", "_time_ahead", "car_ahead"]],
        on=["LapNumber", "Position"],
        how="left",
    )
    merged.index = laps.index
    laps["car_ahead"] = merged["car_ahead"]
    laps["gap_ahead_s"] = _td_seconds(merged["Time"] - merged["_time_ahead"])
    leader_time = laps.groupby("LapNumber")["Time"].transform("min")
    laps["gap_leader_s"] = _td_seconds(laps["Time"] - leader_time)

    X = laps.loc[laps["Time"].notna(), ["Driver", "LapNumber", "Time"]].sort_values(
        "Time", kind="mergesort"
    )
    laps["road_car_ahead"] = X["Driver"].shift(1)
    laps["road_gap_ahead_s"] = _td_seconds(X["Time"].diff())
    laps["road_ahead_lap_diff"] = X["LapNumber"].shift(1) - X["LapNumber"]
    return laps


def _overtake_disabled_laps(rcm: Optional[pd.DataFrame], max_lap: int) -> set:
    """
    Lap numbers on which Overtake Mode was disabled for at least part of
    the lap according to race control: every lap from an OVERTAKE DISABLED
    message up to and including the lap of the next OVERTAKE ENABLED.
    RCM `Lap` is the LEADER's lap, so this is approximate for backmarkers.
    """
    disabled = set()
    if rcm is None or len(rcm) == 0 or "Message" not in rcm:
        return disabled
    msgs = rcm[rcm["Message"].astype(str).str.contains(r"OVERTAKE\s+(?:ENABLED|DISABLED)")]
    msgs = msgs.sort_values("Time", kind="mergesort")
    state_disabled_from = None
    for _, row in msgs.iterrows():
        kind = _OVERTAKE_PATTERN.search(str(row["Message"])).group(1)
        lap = int(row["Lap"])
        if kind == "DISABLED":
            if state_disabled_from is None:
                state_disabled_from = lap
        else:  # ENABLED
            start = state_disabled_from if state_disabled_from is not None else lap
            disabled.update(range(start, lap + 1))
            state_disabled_from = None
    if state_disabled_from is not None:
        disabled.update(range(state_disabled_from, max_lap + 1))
    return disabled


def _add_overtake_enabled(laps: pd.DataFrame, rcm: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    T-001 §2.5 / FIA B7.2.2: Overtake is off on lap 1, under SC (and red),
    and until race control re-enables it after a restart. Combines the RCM
    disabled-lap set with the per-car SC/red flags -- China 2026 shows why
    both are needed: an SC with only an ENABLED message afterwards (no
    DISABLED message when it was deployed).
    """
    max_lap = int(laps["LapNumber"].max()) if len(laps) else 0
    rcm_disabled = _overtake_disabled_laps(rcm, max_lap)
    laps["overtake_enabled"] = ~(
        (laps["LapNumber"] <= 1)
        | laps["has_sc"]
        | laps["has_red"]
        | laps["LapNumber"].isin(rcm_disabled)
    )
    return laps


def _add_weather(laps: pd.DataFrame, weather: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Most recent weather sample at or before each lap's end (`Time`)."""
    cols = {"TrackTemp": "track_temp_c", "AirTemp": "air_temp_c", "Rainfall": "rainfall"}
    if weather is None or len(weather) == 0:
        for c in cols.values():
            laps[c] = np.nan
        return laps
    W = weather[["Time", *cols.keys()]].dropna(subset=["Time"]).sort_values("Time")
    left = laps[["Time"]].reset_index().dropna(subset=["Time"]).sort_values("Time")
    joined = pd.merge_asof(left, W, on="Time", direction="backward").set_index("index")
    for src, dst in cols.items():
        laps[dst] = joined[src].reindex(laps.index)
    return laps


def extract_laps_df(
    session: fastf1.core.Session,
    compound_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Extract lap-by-lap data from a race session, one row per real lap
    (see module docstring for which rows are kept/dropped).

    Columns (T-001 §4; * = new in T-002):
    - driver, driver_name: 3-letter abbreviation (e.g. "NOR"); driver_name
      duplicates it and is kept only for backward compatibility
    - driver_number*, team*
    - lap_number
    - lap_time_seconds: NULLABLE (NaN on most SC/red-flag laps)
    - lap_start_s*, lap_end_s*: LapStartTime / Time in session seconds.
      lap_end_s is always populated -- use it for gaps and race totals.
    - s1_s*, s2_s*, s3_s*: sector times (s1 is NaN on lap 1)
    - compound: SOFT/MEDIUM/HARD/INTERMEDIATE/WET event label (NaN if
      unknown, see tyre_data_missing)
    - compound_c*: Pirelli C-number for this event (C1..C5, INTER/WET
      passthrough), NaN if the event's mapping isn't recorded
    - tire_age: TyreLife (laps on this set, including earlier sessions)
    - fresh_tyre*: FastF1 FreshTyre -- informational only; tire_age is the
      source of truth (5 known disagreements)
    - tyre_data_imputed*, tyre_data_missing*: see _impute_tyre_data
    - ff1_stint*: FastF1's raw Stint (increments on every pit-lane pass)
    - tyre_stint*, stint_id*: stint counter / "<driver>_<n>" that only
      increments on a REAL tyre change
    - stint_start_age*: tire_age on the stint's first lap (-> Stint.start_age)
    - is_tyre_change*: first lap of a new tyre_stint (not set on lap 1)
    - is_pit_pass_no_change*: out-lap of a same-tyre pit-lane pass
    - pit_in_lap, pit_out_lap*: 0/1
    - pit_lane_s*: pit-lane traversal seconds, on the out-lap
    - pit_under_sc*, pit_under_vsc*, pit_under_red*: on the in-lap
    - position*: nullable float
    - car_ahead*, gap_ahead_s*, gap_leader_s*: race-order gaps
    - road_car_ahead*, road_gap_ahead_s*, road_ahead_lap_diff*: on-road gaps
    - track_status*: raw TrackStatus string
    - is_green_flag: TrackStatus == "1" exactly (unchanged semantics)
    - has_yellow*, has_sc*, has_vsc*, has_red*, is_neutralised*
    - overtake_enabled*
    - is_accurate*, deleted*: FastF1 IsAccurate / Deleted
    - track_temp_c*, air_temp_c*, rainfall*: weather at lap end
    """
    laps = session.laps.copy()

    # --- rows that aren't real laps -------------------------------------
    if "FastF1Generated" in laps:
        laps = laps[~laps["FastF1Generated"].fillna(False).astype(bool)]
    dns = _dns_drivers(session)
    if dns:
        laps = laps[~laps["Driver"].isin(dns)]
    laps = laps.sort_values(["Driver", "LapNumber"], kind="mergesort").reset_index(drop=True)

    # --- tyres & stints ----------------------------------------------------
    laps = _impute_tyre_data(laps)
    laps = _assign_tyre_stints(laps)
    laps = _decode_track_status(laps)
    laps = _add_pit_columns(laps)
    laps = _add_gaps(laps)

    try:
        rcm = session.race_control_messages
    except Exception:
        rcm = None
    laps = _add_overtake_enabled(laps, rcm)

    try:
        weather = session.weather_data
    except Exception:
        weather = None
    laps = _add_weather(laps, weather)

    if compound_map is None:
        try:
            compound_map = compound_map_for_event(*_event_year_name(session))
        except Exception:  # no event metadata (e.g. a hand-built test session)
            compound_map = None
    full_map = dict(_WET_PASSTHROUGH)
    if compound_map:
        full_map.update(compound_map)
    laps["compound_c"] = laps["Compound"].map(full_map)

    out = pd.DataFrame(
        {
            "driver": laps["Driver"],
            "driver_name": laps["Driver"],
            "driver_number": laps["DriverNumber"],
            "team": laps["Team"],
            "lap_number": laps["LapNumber"].astype(int),
            "lap_time_seconds": _td_seconds(laps["LapTime"]),
            "lap_start_s": _td_seconds(laps["LapStartTime"]),
            "lap_end_s": _td_seconds(laps["Time"]),
            "s1_s": _td_seconds(laps["Sector1Time"]),
            "s2_s": _td_seconds(laps["Sector2Time"]),
            "s3_s": _td_seconds(laps["Sector3Time"]),
            "compound": laps["Compound"],
            "compound_c": laps["compound_c"],
            "tire_age": laps["TyreLife"],
            "fresh_tyre": laps["FreshTyre"],
            "tyre_data_imputed": laps["tyre_data_imputed"],
            "tyre_data_missing": laps["tyre_data_missing"],
            "ff1_stint": laps["ff1_stint"],
            "tyre_stint": laps["tyre_stint"],
            "stint_id": laps["stint_id"],
            "stint_start_age": laps["stint_start_age"],
            "is_tyre_change": laps["is_tyre_change"],
            "is_pit_pass_no_change": laps["is_pit_pass_no_change"],
            "pit_in_lap": laps["pit_in_lap"],
            "pit_out_lap": laps["pit_out_lap"],
            "pit_lane_s": laps["pit_lane_s"],
            "pit_under_sc": laps["pit_under_sc"],
            "pit_under_vsc": laps["pit_under_vsc"],
            "pit_under_red": laps["pit_under_red"],
            "position": laps["Position"],
            "car_ahead": laps["car_ahead"],
            "gap_ahead_s": laps["gap_ahead_s"],
            "gap_leader_s": laps["gap_leader_s"],
            "road_car_ahead": laps["road_car_ahead"],
            "road_gap_ahead_s": laps["road_gap_ahead_s"],
            "road_ahead_lap_diff": laps["road_ahead_lap_diff"],
            "track_status": laps["track_status"],
            "is_green_flag": laps["is_green_flag"],
            "has_yellow": laps["has_yellow"],
            "has_sc": laps["has_sc"],
            "has_vsc": laps["has_vsc"],
            "has_red": laps["has_red"],
            "is_neutralised": laps["is_neutralised"],
            "overtake_enabled": laps["overtake_enabled"],
            "is_accurate": laps["IsAccurate"],
            "deleted": laps["Deleted"],
            "track_temp_c": laps["track_temp_c"],
            "air_temp_c": laps["air_temp_c"],
            "rainfall": laps["rainfall"],
        }
    )
    return out.reset_index(drop=True)


def extract_pit_stops(session: fastf1.core.Session) -> pd.DataFrame:
    """
    Extract real pit stop durations from a race session.

    FastF1 sets `PitInTime` on the lap a driver crosses the pit entry line
    and `PitOutTime` on the following lap, when they cross the pit exit
    line -- so a stop's duration is the gap between one lap's PitInTime and
    the *next* lap's PitOutTime for that driver. This is the standard
    proxy for total pit-lane time loss (used to calibrate
    race_simulator.py's DEFAULT_PIT_LOSS_SECONDS).

    NOTE (T-002): this still counts every pit-lane passage, including
    same-tyre passes (drive-throughs, Monaco's SC pit-lane passes) and
    red-flag stops. For a cleaner sample use the per-stop table from
    pit_stops_from_laps() on an extract_laps_df() frame, which carries
    is_tyre_change / pit_under_* flags.

    Returns DataFrame with columns:
    - driver: Driver abbreviation
    - lap_number: The in-lap (lap on which PitInTime was recorded)
    - pit_duration_seconds: PitOutTime - PitInTime, in seconds
    """
    laps = session.laps.copy()
    rows = []

    for driver, driver_laps in laps.groupby("Driver"):
        driver_laps = driver_laps.sort_values("LapNumber").reset_index(drop=True)
        for i in range(len(driver_laps) - 1):
            pit_in = driver_laps.loc[i, "PitInTime"]
            pit_out = driver_laps.loc[i + 1, "PitOutTime"]
            if pd.notna(pit_in) and pd.notna(pit_out):
                rows.append(
                    {
                        "driver": driver,
                        "lap_number": driver_laps.loc[i, "LapNumber"],
                        "pit_duration_seconds": (pit_out - pit_in).total_seconds(),
                    }
                )

    return pd.DataFrame(rows, columns=["driver", "lap_number", "pit_duration_seconds"])


def pit_stops_from_laps(laps_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per completed pit-lane passage (in-lap followed by an out-lap),
    built from an extract_laps_df() frame:
    driver, in_lap, out_lap, pit_lane_s, is_tyre_change,
    is_pit_pass_no_change, pit_under_sc, pit_under_vsc, pit_under_red,
    compound_before, compound_after, start_age_after.
    """
    cols = [
        "driver", "in_lap", "out_lap", "pit_lane_s", "is_tyre_change",
        "is_pit_pass_no_change", "pit_under_sc", "pit_under_vsc", "pit_under_red",
        "compound_before", "compound_after", "start_age_after",
    ]
    df = laps_df.sort_values(["driver", "lap_number"])
    nxt = df.groupby("driver").shift(-1)
    is_stop = (df["pit_in_lap"] == 1) & (nxt["pit_out_lap"] == 1) & (
        nxt["lap_number"] == df["lap_number"] + 1
    )
    stops = pd.DataFrame(
        {
            "driver": df["driver"],
            "in_lap": df["lap_number"],
            "out_lap": nxt["lap_number"],
            "pit_lane_s": nxt["pit_lane_s"],
            "is_tyre_change": nxt["is_tyre_change"],
            "is_pit_pass_no_change": nxt["is_pit_pass_no_change"],
            "pit_under_sc": df["pit_under_sc"],
            "pit_under_vsc": df["pit_under_vsc"],
            "pit_under_red": df["pit_under_red"],
            "compound_before": df["compound"],
            "compound_after": nxt["compound"],
            "start_age_after": nxt["stint_start_age"],
        }
    )[is_stop]
    stops["out_lap"] = stops["out_lap"].astype(int)
    return stops[cols].reset_index(drop=True)


def extract_results_df(session: fastf1.core.Session) -> pd.DataFrame:
    """
    Classification sidecar. results.Time is the winner's total race time
    but, for every other lead-lap finisher, the GAP to the winner INCLUDING
    post-race penalties (T-001 §1.2 -- contradicts FastF1's docs), and is
    not a usable gap for Lapped cars. So:
      gap_to_winner_s: 0 for P1, results.Time for other 'Finished' cars,
                       NaN otherwise
      race_time_s:     winner total + gap_to_winner_s (NaN if not on lead lap)
    Score finishing order against `position`, not reconstructed track order.
    """
    r = session.results.copy()
    time_s = _td_seconds(r["Time"])
    is_winner = r["Position"] == 1
    winner_total = time_s[is_winner].iloc[0] if is_winner.any() else np.nan
    gap = pd.Series(np.nan, index=r.index)
    gap[is_winner] = 0.0
    lead_lap = (r["Status"] == "Finished") & ~is_winner
    gap[lead_lap] = time_s[lead_lap]
    return pd.DataFrame(
        {
            "driver": r["Abbreviation"],
            "driver_number": r["DriverNumber"],
            "team": r["TeamName"],
            "grid_position": r["GridPosition"],
            "position": r["Position"],
            "classified_position": r["ClassifiedPosition"],
            "status": r["Status"],
            "laps": r["Laps"],
            "points": r["Points"],
            "gap_to_winner_s": gap,
            "race_time_s": winner_total + gap,
            "q1_s": _td_seconds(r["Q1"]),
            "q2_s": _td_seconds(r["Q2"]),
            "q3_s": _td_seconds(r["Q3"]),
        }
    ).reset_index(drop=True)


def extract_weather_df(session: fastf1.core.Session) -> pd.DataFrame:
    """Raw session.weather_data with Time -> time_s (session seconds)."""
    w = session.weather_data.copy()
    w.insert(0, "time_s", _td_seconds(w.pop("Time")))
    w.columns = [c if c == "time_s" else re.sub(r"(?<!^)([A-Z])", r"_\1", c).lower() for c in w.columns]
    return w.reset_index(drop=True)


def extract_track_status_df(session: fastf1.core.Session) -> pd.DataFrame:
    """session.track_status intervals: time_s (session seconds), status, message."""
    ts = session.track_status
    return pd.DataFrame(
        {"time_s": _td_seconds(ts["Time"]), "status": ts["Status"], "message": ts["Message"]}
    ).reset_index(drop=True)


def extract_rcm_df(session: fastf1.core.Session) -> pd.DataFrame:
    """
    Race-control messages: SafetyCar/Flag categories plus Overtake, pit-lane
    and penalty texts. `time_utc` is WALL CLOCK (not session time); use
    `lap` (leader's lap) to align with laps.
    """
    rcm = session.race_control_messages
    keep = rcm["Category"].isin(["SafetyCar", "Flag"]) | rcm["Message"].astype(str).str.contains(
        _RCM_KEEP_PATTERN
    )
    r = rcm[keep]
    return pd.DataFrame(
        {
            "time_utc": r["Time"].astype(str),
            "lap": r["Lap"],
            "category": r["Category"],
            "message": r["Message"],
            "status": r["Status"],
            "flag": r["Flag"],
            "scope": r["Scope"],
            "sector": r["Sector"],
            "racing_number": r["RacingNumber"],
        }
    ).reset_index(drop=True)


def _event_year_name(session) -> Tuple[int, str]:
    event = session.event
    return int(pd.Timestamp(event["EventDate"]).year), str(event.get("EventName", "Unknown"))


def get_race_tables(year: int, gp: str) -> dict:
    """
    Load one race and return every table T-002 extracts:
    {"laps", "results", "weather", "track_status", "rcm", "pit_stops",
     "race_info"}. Sidecar extractors that fail (e.g. a table missing from
    an old cache) yield None rather than aborting the whole race.
    """
    session = load_race_session(year, gp, session_type="R")
    _, event_name = _event_year_name(session)
    compound_map = compound_map_for_event(year, event_name)
    if compound_map is None:
        print(f"  WARNING: no C-compound mapping recorded for {year} {event_name}; compound_c will be NaN")
    laps_df = extract_laps_df(session, compound_map=compound_map)

    tables = {"laps": laps_df, "pit_stops": pit_stops_from_laps(laps_df)}
    for name, fn in (
        ("results", extract_results_df),
        ("weather", extract_weather_df),
        ("track_status", extract_track_status_df),
        ("rcm", extract_rcm_df),
    ):
        try:
            tables[name] = fn(session)
        except Exception as e:  # keep the laps even if a sidecar is unavailable
            print(f"  WARNING: {name} sidecar unavailable: {e}")
            tables[name] = None

    event = session.event
    tables["race_info"] = {
        "year": year,
        "gp": gp,
        "event_name": event_name,
        "circuit": event.get("Location", "Unknown"),
        "event_format": event.get("EventFormat", "Unknown"),
        "event_date": str(pd.Timestamp(event["EventDate"]).date()),
        "total_laps": int(session.total_laps) if getattr(session, "total_laps", None) else None,
        "num_laps": int(laps_df["lap_number"].max()),
        "num_drivers": int(laps_df["driver"].nunique()),
        "compound_map": compound_map,
        "compound_map_source": COMPOUND_ALLOCATION_SOURCE.get(
            (year, event_name), "T-001 research note §3.2 (team/Pirelli previews)"
        ) if compound_map else None,
    }
    return tables


def save_race_tables(tables: dict, out_dir: str, gp: str, year: int) -> Dict[str, str]:
    """
    Write get_race_tables() output as {gp}_{year}_<table>.csv files plus
    {gp}_{year}_race_info.json. Only `*_laps.csv` is read by the tire
    model's glob, so sidecars never leak into fitting by accident.
    """
    os.makedirs(out_dir, exist_ok=True)
    written = {}
    for name in ("laps", "results", "weather", "track_status", "rcm", "pit_stops"):
        df = tables.get(name)
        if df is None:
            continue
        path = os.path.join(out_dir, f"{gp}_{year}_{name}.csv")
        df.to_csv(path, index=False)
        written[name] = path
    path = os.path.join(out_dir, f"{gp}_{year}_race_info.json")
    with open(path, "w") as f:
        json.dump(tables["race_info"], f, indent=2, default=str)
    written["race_info"] = path
    return written


def get_race_data(year: int, gp: str) -> Tuple[pd.DataFrame, dict]:
    """
    Convenience function: load, extract, and return race data + metadata.
    (Backward-compatible wrapper; use get_race_tables for the sidecars.)

    Returns:
        (laps_df, race_info_dict)
    """
    tables = get_race_tables(year, gp)
    return tables["laps"], tables["race_info"]
