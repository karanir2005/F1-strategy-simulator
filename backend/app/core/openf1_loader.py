"""
OpenF1 (+ Jolpica) -> per-lap CSV in the same schema data_loader.py writes
from FastF1 (T-009, 2026-10-19).

WHY: F1's live-timing archive (FastF1's source) now needs a paid login, so
new races can't be pulled through FastF1. OpenF1 (api.openf1.org/v1, free,
no auth) still serves historical sessions, and Jolpica (api.jolpi.ca/ergast)
serves results / grid / qualifying.

LAYERS
  fetch   : cached GETs (data/openf1/raw/<session_key>/<endpoint>.json and
            data/openf1/raw/jolpica/...). Reruns never refetch. ~2.5 req/s,
            exponential back-off on 429 / 5xx.
  convert : pure functions raw JSON -> DataFrames (testable offline).
  build   : build_race(...) -> {"laps", "results", "quali", "weather",
            "track_status", "rcm", "pit_stops", "race_info"};
            write_race() saves them in data/staging/ using the file names
            data/processed/ uses ({gp}_{year}_<table>.csv), plus `_quali.csv`.

SCHEMA = data_loader.extract_laps_df's schema, plus `season`, `era` and
`source` ("openf1") on every laps row (sidecars get season/era/source too).
Stint / pit / gap / overtake logic is NOT re-implemented: the FastF1-named
frame is handed to data_loader's own helpers (_assign_tyre_stints,
_decode_track_status, _add_gaps, _add_overtake_enabled, _add_weather) so
both sources share one definition of "real tyre change", gaps, etc.

KEY CONVENTIONS (verified against the 10 FastF1 races; numbers in the T-009
handoff):
  * OpenF1 stints.tyre_age_at_start = laps already on the set BEFORE the
    stint. FastF1 TyreLife on the first lap = that + 1. So
    tire_age(lap) = tyre_age_at_start + (lap - lap_start) + 1.
  * Session-time columns (lap_start_s, lap_end_s, weather/track_status
    time_s) are seconds since the earliest lap-1 `date_start` (race lights
    out), NOT FastF1 session time (which is not recoverable). Only
    differences are meaningful, which is all the gap code uses.
  * Lap 1 occasionally has no lap_duration (standing start); it is filled
    from next-lap timing (the only lap where we do that).
  * TrackStatus per lap is rebuilt from race_control: SC (deployed -> track
    clear), VSC (deployed -> ending -> clear), red (flag -> restart), yellow
    (sector yellow/double yellow until that sector clears), then intersected
    with each lap's own [start, end] window and written as a FastF1-style
    code string ("1", "12", "124", "67", "5" ...).
  * `deleted` is parsed from "CAR n (XXX) LAP DELETED ... LAP k" messages.
    OpenF1 has no IsAccurate, so `is_accurate` is NaN (nothing reads it).
  * compound_c uses data_loader.COMPOUND_ALLOCATION where recorded (the 10
    original 2026 events only); NaN otherwise -- never guessed.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

from . import data_loader as dl

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OPENF1 = "https://api.openf1.org/v1"
JOLPICA = "https://api.jolpi.ca/ergast/f1"
RAW_DIR = os.path.join(REPO_ROOT, "data", "openf1", "raw")
STAGING_DIR = os.path.join(REPO_ROOT, "data", "staging")
MIN_INTERVAL_S = 0.4  # ~2.5 req/s

ERA_BY_SEASON = {2025: "2025_regs", 2026: "2026_regs"}

# OpenF1 circuit_short_name -> file label (existing 10 files keep their names).
CIRCUIT_LABEL = {
    "Melbourne": "Australia", "Shanghai": "China", "Suzuka": "Japan", "Sakhir": "Bahrain",
    "Jeddah": "Saudi", "Miami": "Miami", "Imola": "Imola", "Monte Carlo": "Monaco",
    "Catalunya": "Barcelona", "Montreal": "Canada", "Spielberg": "Austria",
    "Silverstone": "Britain", "Spa-Francorchamps": "Belgium", "Hungaroring": "Hungary",
    "Zandvoort": "Netherlands", "Monza": "Italy", "Baku": "Azerbaijan",
    "Singapore": "Singapore", "Austin": "Austin", "Mexico City": "Mexico",
    "Interlagos": "Brazil", "Las Vegas": "LasVegas", "Lusail": "Qatar",
    "Yas Marina Circuit": "AbuDhabi", "Madring": "Madrid", "Kuala Lumpur": "Malaysia",
}
# OpenF1 mangles accents in a few location strings; labels key on the ASCII
# circuit_short_name which is stable.

_last_request = [0.0]


# ---------------------------------------------------------------- fetch ----
def _get_json(url: str, params: Optional[dict] = None, retries: int = 8):
    delay = 2.0
    for _ in range(retries):
        wait = MIN_INTERVAL_S - (time.time() - _last_request[0])
        if wait > 0:
            time.sleep(wait)
        _last_request[0] = time.time()
        try:
            r = requests.get(url, params=params, timeout=90)
        except requests.RequestException:
            time.sleep(delay)
            delay = min(delay * 2, 60)
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code == 404:
            # OpenF1 returns 404 {"detail":"No results found."} for empty queries
            return []
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(delay)
            delay = min(delay * 2, 60)
            continue
        r.raise_for_status()
    raise RuntimeError(f"GET {url} {params} failed after {retries} attempts")


def cached_get(rel_path: str, url: str, params: Optional[dict] = None, raw_dir: str = RAW_DIR):
    """GET with an on-disk JSON cache keyed by rel_path (never refetches)."""
    path = os.path.join(raw_dir, rel_path)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    data = _get_json(url, params)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)
    return data


SESSION_ENDPOINTS = ["drivers", "laps", "stints", "pit", "race_control", "weather", "intervals",
                     "position", "session_result", "starting_grid"]


def fetch_session(session_key: int, raw_dir: str = RAW_DIR) -> Dict[str, list]:
    return {ep: cached_get(f"{session_key}/{ep}.json", f"{OPENF1}/{ep}",
                           {"session_key": session_key}, raw_dir) for ep in SESSION_ENDPOINTS}


def fetch_sessions(year: int, session_name: str, raw_dir: str = RAW_DIR) -> list:
    return cached_get(f"sessions_{year}_{session_name}.json", f"{OPENF1}/sessions",
                      {"year": year, "session_name": session_name}, raw_dir)


def fetch_jolpica(season: int, rnd: int, what: str, raw_dir: str = RAW_DIR) -> dict:
    """what in {results, qualifying, pitstops}."""
    assert what in ("results", "qualifying", "pitstops")
    return cached_get(f"jolpica/{season}_{rnd}_{what}.json",
                      f"{JOLPICA}/{season}/{rnd}/{what}.json", {"limit": 100}, raw_dir)


def fetch_jolpica_schedule(season: int, raw_dir: str = RAW_DIR) -> list:
    d = cached_get(f"jolpica/schedule_{season}.json", f"{JOLPICA}/{season}.json", {"limit": 100}, raw_dir)
    return d["MRData"]["RaceTable"]["Races"]


def list_races(seasons=(2025, 2026), today: Optional[str] = None, raw_dir: str = RAW_DIR) -> List[dict]:
    """
    Held, non-cancelled races: OpenF1 Race session joined to the Jolpica
    round by calendar date. Returns dicts with season, round, session_key,
    label, race_name, circuit, date. Sorted by date.
    """
    today = today or time.strftime("%Y-%m-%d")
    out = []
    for season in seasons:
        sched = {r["date"]: r for r in fetch_jolpica_schedule(season, raw_dir)}
        for s in fetch_sessions(season, "Race", raw_dir):
            day = s["date_start"][:10]
            if s.get("is_cancelled") or day > today or day not in sched:
                continue
            r = sched[day]
            out.append({
                "season": season, "round": int(r["round"]), "session_key": s["session_key"],
                "label": CIRCUIT_LABEL.get(s["circuit_short_name"], re.sub(r"\W+", "", s["circuit_short_name"])),
                "race_name": r["raceName"], "circuit": s["circuit_short_name"], "date": day,
            })
    return sorted(out, key=lambda d: d["date"])


# -------------------------------------------------------------- convert ----
def _ts(x) -> pd.Series:
    return pd.to_datetime(x, utc=True, format="ISO8601")


def _drivers_table(raw_drivers: list) -> pd.DataFrame:
    d = pd.DataFrame(raw_drivers)
    return d[["driver_number", "name_acronym", "team_name"]].drop_duplicates("driver_number")


def build_status_timeline(rcm: pd.DataFrame, t_end: pd.Timestamp) -> pd.DataFrame:
    """
    Piecewise-constant track status from race control. Returns rows
    (start, end, code) covering [first message, t_end], code in
    {"1","2","4","5","6","7"}; priority red > SC > VSC > yellow > clear.
    """
    if rcm.empty:
        return pd.DataFrame(columns=["start", "end", "code"])
    r = rcm.sort_values("date", kind="mergesort")
    msg = r["message"].astype(str).str.upper()
    t0 = r["date"].min()
    ivs = {"4": [], "5": [], "6": [], "7": [], "2": []}  # code -> [(a, b)]
    # events that end an SC/VSC/red window
    clears = r.loc[(r["flag"].astype(str).isin(["CLEAR", "GREEN"]) & (r["scope"] == "Track"))
                   | msg.str.contains("SESSION STARTED") | msg.str.contains("SESSION FINISHED"), "date"].tolist()

    def next_clear(t, strict_after=True):
        for c in clears:
            if c > t:
                return c
        return t_end

    open_sc = open_vsc = open_red = None
    vsc_ending = None
    for t, m, fl in zip(r["date"], msg, r["flag"].astype(str)):
        if "RED FLAG" in m and ("SUSPENDED" in m or m.strip() == "RED FLAG") and "INFRINGEMENT" not in m:
            ivs["5"].append((t, _restart_after(r, t, t_end)))
        elif m.strip() == "SAFETY CAR DEPLOYED":
            ivs["4"].append((t, next_clear(t)))
        elif m.strip() in ("VSC DEPLOYED", "VIRTUAL SAFETY CAR DEPLOYED"):
            end = next_clear(t)
            ending = r.loc[(r["date"] > t) & (r["date"] <= end)
                           & msg.isin(["VSC ENDING", "VIRTUAL SAFETY CAR ENDING"]), "date"]
            if len(ending):
                ivs["6"].append((t, ending.iloc[0]))
                ivs["7"].append((ending.iloc[0], end))
            else:
                ivs["6"].append((t, end))
    # yellows: sector yellow/double yellow until that sector clears
    sect = r[(r["scope"] == "Sector") & r["flag"].astype(str).isin(["YELLOW", "DOUBLE YELLOW", "CLEAR"])]
    open_y = {}
    for t, fl, sec in zip(sect["date"], sect["flag"].astype(str), sect["sector"]):
        if fl in ("YELLOW", "DOUBLE YELLOW"):
            open_y.setdefault(sec, t)
        elif sec in open_y:
            ivs["2"].append((open_y.pop(sec), t))
    for sec, t in open_y.items():
        ivs["2"].append((t, next_clear(t)))

    cuts = {t0, t_end}
    for lst in ivs.values():
        for a, b in lst:
            cuts.update([max(a, t0), min(b, t_end)])
    cuts = sorted(c for c in cuts if t0 <= c <= t_end)
    rows = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        mid = a + (b - a) / 2
        code = "1"
        for c in ("5", "4", "6", "7", "2"):  # priority order
            if any(x <= mid < y for x, y in ivs[c]):
                code = c
                break
        if rows and rows[-1][2] == code and rows[-1][1] == a:
            rows[-1] = (rows[-1][0], b, code)
        else:
            rows.append((a, b, code))
    return pd.DataFrame(rows, columns=["start", "end", "code"])


def _restart_after(r: pd.DataFrame, t, t_end):
    """Red flag ends at the next SESSION STARTED (race resumed), else t_end."""
    m = r.loc[(r["date"] > t) & r["message"].astype(str).str.upper().str.contains("SESSION STARTED"), "date"]
    return m.iloc[0] if len(m) else t_end


def lap_track_status(starts: pd.Series, ends: pd.Series, tl: pd.DataFrame) -> pd.Series:
    """FastF1-style TrackStatus string for each [start, end] lap window."""
    out = []
    ta = tl["start"].values
    tb = tl["end"].values
    codes = tl["code"].values
    for s, e in zip(starts.values, ends.values):
        if pd.isna(s) or pd.isna(e) or len(tl) == 0:
            out.append(np.nan)
            continue
        m = (ta < e) & (tb > s)
        seq = []
        for c in codes[m]:
            if not seq or seq[-1] != c:
                seq.append(c)
        out.append("".join(seq) if seq else "1")
    return pd.Series(out, index=starts.index, dtype=object)


_DELETED_RE = re.compile(r"CAR\s+(\d+)\s+\(\w+\)\s+LAP DELETED.*?LAP\s+(\d+)", re.I)


def deleted_laps(rcm: pd.DataFrame) -> set:
    out = set()
    for m in rcm["message"].astype(str):
        g = _DELETED_RE.search(m)
        if g:
            out.add((int(g.group(1)), int(g.group(2))))
    return out


def normalise_stints(stints: pd.DataFrame, pit: pd.DataFrame, last_lap: Dict[int, int]) -> pd.DataFrame:
    """
    Two OpenF1 stint quirks found by diffing against FastF1 (T-009):
      1. Some races (Canada, Monaco, Hungary, ...) start stint k>1 on the
         IN-lap instead of the out-lap (stint.lap_start == the pit table's
         lap_number for that driver). That makes every later tire_age one
         too high and the stint boundary a lap early. If a stint starts on
         a pit in-lap, shift lap_start (and the previous lap_end) by +1.
         (Japan/China already start on the out-lap; the rule leaves them
         untouched.)
      2. A driver's final stint is sometimes a 1-lap row with compound=None
         (the retirement / in-lap). Merge it into the previous stint so it
         doesn't create a phantom tyre change. Mid-race null-compound
         stints are left alone (genuinely unknown -> tyre_data_missing).
    """
    if stints.empty:
        return stints
    st = stints.sort_values(["driver_number", "stint_number"]).reset_index(drop=True).copy()
    in_laps = set()
    if len(pit):
        in_laps = set(zip(pit["driver_number"], pit["lap_number"]))
    drop = []
    for dn, g in st.groupby("driver_number"):
        idx = list(g.index)
        for j in range(1, len(idx)):
            i, prev = idx[j], idx[j - 1]
            ls = st.at[i, "lap_start"]
            if (dn, ls) in in_laps:
                st.at[i, "lap_start"] = ls + 1
                st.at[prev, "lap_end"] = ls
        last = idx[-1]
        if len(idx) > 1 and (st.at[last, "compound"] is None or pd.isna(st.at[last, "compound"])):
            le = st.at[last, "lap_end"]
            le = last_lap.get(dn, st.at[last, "lap_start"]) if pd.isna(le) else le
            if le - st.at[last, "lap_start"] <= 1:
                st.at[idx[-2], "lap_end"] = np.nan  # open-ended: covers the final lap(s)
                drop.append(last)
    return st.drop(index=drop).reset_index(drop=True)


def stints_to_laps(laps: pd.DataFrame, stints: pd.DataFrame) -> pd.DataFrame:
    """
    Attach Stint / Compound / TyreLife / FreshTyre to each (driver_number,
    lap_number). A lap belongs to the stint with the greatest lap_start <=
    lap, provided lap <= lap_end (null lap_end = open-ended). Laps with no
    stint, or a null compound, keep Compound NaN (-> tyre_data_missing).
    """
    laps = laps.copy()
    for c in ("Stint", "Compound", "TyreLife", "FreshTyre"):
        laps[c] = np.nan
    laps["Compound"] = laps["Compound"].astype(object)
    laps["FreshTyre"] = laps["FreshTyre"].astype(object)
    if stints.empty:
        return laps
    st = stints.sort_values(["driver_number", "lap_start"])
    for dn, g in laps.groupby("driver_number"):
        sg = st[st["driver_number"] == dn]
        if sg.empty:
            continue
        starts = sg["lap_start"].values
        idx = np.searchsorted(starts, g["lap_number"].values, side="right") - 1
        for (ix, lap), k in zip(g["lap_number"].items(), idx):
            if k < 0:
                continue
            row = sg.iloc[k]
            if pd.notna(row["lap_end"]) and lap > row["lap_end"]:
                continue
            laps.at[ix, "Stint"] = row["stint_number"]
            comp = row["compound"]
            if comp is not None and str(comp).upper() not in ("UNKNOWN", "NONE", "NAN", ""):
                laps.at[ix, "Compound"] = str(comp).upper()
                age0 = row["tyre_age_at_start"]
                if pd.notna(age0):
                    laps.at[ix, "TyreLife"] = float(age0) + (lap - row["lap_start"]) + 1
                    laps.at[ix, "FreshTyre"] = bool(age0 == 0)
    return laps


def _dedupe_positions(L: pd.DataFrame) -> pd.Series:
    """
    OpenF1 position is an event stream sampled at lap end, so two cars can
    momentarily share a position (a swap mid-update). data_loader._add_gaps
    needs one car per (lap, position) -- FastF1 guarantees that. In any lap
    with a duplicate, re-rank that lap's cars by (position, lap-end time).
    """
    pos = L["Position"].copy()
    for lap, g in L.groupby("lap_number"):
        p = g["Position"].dropna()
        if p.duplicated().any():
            order = pd.DataFrame({"p": p, "t": g.loc[p.index, "_end"]}).sort_values(["p", "t"], kind="mergesort")
            pos.loc[order.index] = np.arange(1, len(order) + 1, dtype=float)
    return pos


def convert_laps(raw: Dict[str, list], season: int, compound_map: Optional[dict] = None,
                 result_laps: Optional[Dict[str, tuple]] = None):
    """
    Raw OpenF1 session dict (fetch_session output) -> (laps_df, aux) where
    aux = {"t0": race-start Timestamp, "rcm": DataFrame, "weather": ...,
    "timeline": status timeline, "trimmed": n post-race rows dropped}.
    The laps frame follows extract_laps_df.

    result_laps: {acronym: (laps_completed, status)} from Jolpica. OpenF1
    keeps one extra, untimed row after a car's last completed lap (the lap
    it retired on, or the in-lap after the chequered flag); FastF1 has none,
    so rows with lap_number > laps_completed AND no lap_duration are dropped.
    """
    drivers = _drivers_table(raw["drivers"])
    L = pd.DataFrame(raw["laps"])
    L = L.merge(drivers, on="driver_number", how="left")
    # drop DNS drivers (FastF1 loader does the same via results Status)
    dns = {r["driver_number"] for r in raw.get("session_result", []) if r.get("dns")}
    L = L[~L["driver_number"].isin(dns)].copy()
    L["date_start"] = _ts(L["date_start"])
    trimmed = 0
    if result_laps:
        lim = L["name_acronym"].map(lambda a: result_laps.get(a, (None, ""))[0])
        bad = lim.notna() & (L["lap_number"] > lim) & L["lap_duration"].isna()
        trimmed = int(bad.sum())
        L = L[~bad].copy()
    L = L.sort_values(["driver_number", "lap_number"], kind="mergesort").reset_index(drop=True)
    t0 = L.loc[L["lap_number"] == 1, "date_start"].min()

    nxt_start = L.groupby("driver_number")["date_start"].shift(-1)
    # a "next lap" row only counts if it is lap+1
    nxt_lap = L.groupby("driver_number")["lap_number"].shift(-1)
    nxt_start = nxt_start.where(nxt_lap == L["lap_number"] + 1)
    dur = pd.to_timedelta(L["lap_duration"], unit="s")
    L["_end"] = L["date_start"] + dur
    L["_end"] = L["_end"].fillna(nxt_start)
    # lap 1: fill duration from the lap-2 start when absent
    fill1 = (L["lap_number"] == 1) & L["lap_duration"].isna() & L["_end"].notna()
    L.loc[fill1, "lap_duration"] = (L.loc[fill1, "_end"] - L.loc[fill1, "date_start"]).dt.total_seconds()

    st = pd.DataFrame(raw["stints"])
    pit = pd.DataFrame(raw["pit"])
    st = normalise_stints(st, pit, L.groupby("driver_number")["lap_number"].max().to_dict())
    L = stints_to_laps(L, st)

    # rcm / track status
    rcm = pd.DataFrame(raw["race_control"])
    if not rcm.empty:
        rcm["date"] = _ts(rcm["date"])
    t_end = L["_end"].max()
    timeline = build_status_timeline(rcm, t_end) if not rcm.empty else pd.DataFrame(columns=["start", "end", "code"])
    L["TrackStatus"] = lap_track_status(L["date_start"], L["_end"], timeline)

    # pits
    L["pit_in_lap"] = 0
    L["pit_lane_prev"] = np.nan
    if not pit.empty:
        pit = pit.sort_values("date").drop_duplicates(["driver_number", "lap_number"], keep="first")
        key = set(zip(pit["driver_number"], pit["lap_number"]))
        L["pit_in_lap"] = [int((d, n) in key) for d, n in zip(L["driver_number"], L["lap_number"])]
        lane = {(d, n + 1): v for d, n, v in zip(pit["driver_number"], pit["lap_number"],
                                                 pit["lane_duration"].fillna(pit["pit_duration"]))}
        L["pit_lane_prev"] = [lane.get((d, n), np.nan) for d, n in zip(L["driver_number"], L["lap_number"])]
    L["pit_out_lap"] = L["is_pit_out_lap"].fillna(False).astype(int)

    # position at each lap end
    pos = pd.DataFrame(raw["position"])
    L["Position"] = np.nan
    if not pos.empty:
        pos["date"] = _ts(pos["date"])
        pos = pos.sort_values("date")
        left = L[["driver_number", "_end"]].dropna().sort_values("_end")
        m = pd.merge_asof(left, pos[["date", "driver_number", "position"]], left_on="_end", right_on="date",
                          by="driver_number", direction="backward")
        L.loc[left.index, "Position"] = m["position"].values  # merge_asof keeps left order == sorted left
        L["Position"] = _dedupe_positions(L)

    # weather
    wx = pd.DataFrame(raw["weather"])
    if not wx.empty:
        wx["date"] = _ts(wx["date"])
        wx = wx.sort_values("date")
        weather = pd.DataFrame({
            "Time": wx["date"] - t0, "TrackTemp": wx["track_temperature"], "AirTemp": wx["air_temperature"],
            "Rainfall": wx["rainfall"].astype(bool), "Humidity": wx["humidity"], "Pressure": wx["pressure"],
            "WindDirection": wx["wind_direction"], "WindSpeed": wx["wind_speed"]})
    else:
        weather = None

    # ---- assemble using data_loader's helpers (FastF1 column names) ------
    F = pd.DataFrame({
        "Driver": L["name_acronym"], "DriverNumber": L["driver_number"].astype(str), "Team": L["team_name"],
        "LapNumber": L["lap_number"], "Stint": L["Stint"], "Compound": L["Compound"], "TyreLife": L["TyreLife"],
        "FreshTyre": L["FreshTyre"], "TrackStatus": L["TrackStatus"], "Position": L["Position"],
        "Time": L["_end"] - t0, "LapStartTime": L["date_start"] - t0,
        "LapTime": pd.to_timedelta(L["lap_duration"], unit="s"),
        "Sector1Time": pd.to_timedelta(L["duration_sector_1"], unit="s"),
        "Sector2Time": pd.to_timedelta(L["duration_sector_2"], unit="s"),
        "Sector3Time": pd.to_timedelta(L["duration_sector_3"], unit="s"),
    })
    F = dl._impute_tyre_data(F)
    F = dl._assign_tyre_stints(F)
    F = dl._decode_track_status(F)
    F["pit_in_lap"] = L["pit_in_lap"].values
    F["pit_out_lap"] = L["pit_out_lap"].values
    F["pit_lane_s"] = np.where(L["pit_out_lap"] == 1, L["pit_lane_prev"], np.nan)
    is_in = F["pit_in_lap"] == 1
    F["pit_under_sc"] = is_in & F["has_sc"]
    F["pit_under_vsc"] = is_in & F["has_vsc"]
    F["pit_under_red"] = is_in & F["has_red"]
    F = dl._add_gaps(F)
    rcm_for_ot = pd.DataFrame({"Message": rcm["message"], "Lap": rcm["lap_number"], "Time": rcm["date"]}) \
        if not rcm.empty else None
    F = dl._add_overtake_enabled(F, rcm_for_ot)
    F = dl._add_weather(F, weather)

    full_map = dict(dl._WET_PASSTHROUGH)
    if compound_map:
        full_map.update(compound_map)
    F["compound_c"] = F["Compound"].map(full_map)

    dele = deleted_laps(rcm) if not rcm.empty else set()
    deleted = [(int(n), int(k)) in dele for n, k in zip(F["DriverNumber"], F["LapNumber"])]

    era = ERA_BY_SEASON.get(season, f"{season}_regs")
    out = pd.DataFrame({
        "driver": F["Driver"], "driver_name": F["Driver"], "driver_number": F["DriverNumber"],
        "team": F["Team"], "lap_number": F["LapNumber"].astype(int),
        "lap_time_seconds": dl._td_seconds(F["LapTime"]),
        "lap_start_s": dl._td_seconds(F["LapStartTime"]), "lap_end_s": dl._td_seconds(F["Time"]),
        "s1_s": dl._td_seconds(F["Sector1Time"]), "s2_s": dl._td_seconds(F["Sector2Time"]),
        "s3_s": dl._td_seconds(F["Sector3Time"]),
        "compound": F["Compound"], "compound_c": F["compound_c"], "tire_age": F["TyreLife"],
        "fresh_tyre": F["FreshTyre"], "tyre_data_imputed": F["tyre_data_imputed"],
        "tyre_data_missing": F["tyre_data_missing"], "ff1_stint": F["ff1_stint"],
        "tyre_stint": F["tyre_stint"], "stint_id": F["stint_id"], "stint_start_age": F["stint_start_age"],
        "is_tyre_change": F["is_tyre_change"], "is_pit_pass_no_change": F["is_pit_pass_no_change"],
        "pit_in_lap": F["pit_in_lap"], "pit_out_lap": F["pit_out_lap"], "pit_lane_s": F["pit_lane_s"],
        "pit_under_sc": F["pit_under_sc"], "pit_under_vsc": F["pit_under_vsc"],
        "pit_under_red": F["pit_under_red"], "position": F["Position"], "car_ahead": F["car_ahead"],
        "gap_ahead_s": F["gap_ahead_s"], "gap_leader_s": F["gap_leader_s"],
        "road_car_ahead": F["road_car_ahead"], "road_gap_ahead_s": F["road_gap_ahead_s"],
        "road_ahead_lap_diff": F["road_ahead_lap_diff"], "track_status": F["track_status"],
        "is_green_flag": F["is_green_flag"], "has_yellow": F["has_yellow"], "has_sc": F["has_sc"],
        "has_vsc": F["has_vsc"], "has_red": F["has_red"], "is_neutralised": F["is_neutralised"],
        "overtake_enabled": F["overtake_enabled"], "is_accurate": np.nan, "deleted": deleted,
        "track_temp_c": F["track_temp_c"], "air_temp_c": F["air_temp_c"], "rainfall": F["rainfall"],
        "season": season, "era": era, "source": "openf1",
    })
    out = out.sort_values(["driver", "lap_number"], kind="mergesort").reset_index(drop=True)
    aux = {"t0": t0, "rcm": rcm, "weather": weather, "timeline": timeline, "pit": pit, "trimmed": trimmed}
    return out, aux


# ---- sidecars ---------------------------------------------------------------
def _lap_str_to_s(x) -> float:
    """'1:30.035' -> 90.035; '' / None -> NaN."""
    if not x:
        return np.nan
    m = re.fullmatch(r"(?:(\d+):)?(\d+(?:\.\d+)?)", str(x).strip())
    if not m:
        return np.nan
    return (int(m.group(1)) * 60 if m.group(1) else 0) + float(m.group(2))


def convert_quali(jolpica_quali: dict, season: int) -> pd.DataFrame:
    races = jolpica_quali["MRData"]["RaceTable"]["Races"]
    rows = []
    for q in (races[0]["QualifyingResults"] if races else []):
        rows.append({
            "driver": q["Driver"]["code"], "driver_number": q["number"], "team": q["Constructor"]["name"],
            "quali_position": int(q["position"]), "q1_s": _lap_str_to_s(q.get("Q1")),
            "q2_s": _lap_str_to_s(q.get("Q2")), "q3_s": _lap_str_to_s(q.get("Q3")),
        })
    df = pd.DataFrame(rows)
    if len(df):
        df["season"] = season
        df["era"] = ERA_BY_SEASON.get(season, f"{season}_regs")
        df["source"] = "jolpica"
    return df


def convert_results(jolpica_results: dict, quali: pd.DataFrame, season: int) -> pd.DataFrame:
    """
    Same columns as the FastF1 results sidecar. gap_to_winner_s / race_time_s
    come from Jolpica Time.millis for 'Finished' cars on the winner's lap
    (millis is the total race time); others NaN. `status` is Jolpica's
    vocabulary ('Finished', 'Lapped', 'Retired', 'Accident', ...) and
    differs from FastF1's -- see handoff.
    """
    races = jolpica_results["MRData"]["RaceTable"]["Races"]
    rows = []
    for r in (races[0]["Results"] if races else []):
        ms = (r.get("Time") or {}).get("millis")
        rows.append({
            "driver": r["Driver"]["code"], "driver_number": r["number"], "team": r["Constructor"]["name"],
            "grid_position": float(r["grid"]), "position": float(r["position"]),
            "classified_position": r["positionText"] if r["positionText"].isdigit() else r["positionText"],
            "status": r["status"], "laps": float(r["laps"]), "points": float(r["points"]),
            "_ms": float(ms) / 1000 if ms else np.nan,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    win = df.loc[df["position"] == 1, "_ms"]
    wt = win.iloc[0] if len(win) else np.nan
    ok = df["status"].eq("Finished") | (df["position"] == 1)
    df["race_time_s"] = np.where(ok, df["_ms"], np.nan)
    df["gap_to_winner_s"] = np.where(ok, df["race_time_s"] - wt, np.nan)
    df = df.drop(columns="_ms")
    if len(quali):
        df = df.merge(quali[["driver", "q1_s", "q2_s", "q3_s"]], on="driver", how="left")
    else:
        for c in ("q1_s", "q2_s", "q3_s"):
            df[c] = np.nan
    df["season"] = season
    df["era"] = ERA_BY_SEASON.get(season, f"{season}_regs")
    df["source"] = "jolpica"
    cols = ["driver", "driver_number", "team", "grid_position", "position", "classified_position", "status",
            "laps", "points", "gap_to_winner_s", "race_time_s", "q1_s", "q2_s", "q3_s", "season", "era", "source"]
    return df[cols]


def convert_weather(aux: dict, season: int) -> Optional[pd.DataFrame]:
    w = aux["weather"]
    if w is None:
        return None
    return pd.DataFrame({
        "time_s": dl._td_seconds(w["Time"]), "air_temp": w["AirTemp"], "humidity": w["Humidity"],
        "pressure": w["Pressure"], "rainfall": w["Rainfall"], "track_temp": w["TrackTemp"],
        "wind_direction": w["WindDirection"], "wind_speed": w["WindSpeed"], "season": season,
        "era": ERA_BY_SEASON.get(season, f"{season}_regs"), "source": "openf1",
    }).reset_index(drop=True)


_CODE_MSG = {"1": "AllClear", "2": "Yellow", "4": "SCDeployed", "5": "Red", "6": "VSCDeployed", "7": "VSCEnding"}


def convert_track_status(aux: dict, season: int) -> pd.DataFrame:
    tl = aux["timeline"]
    t0 = aux["t0"]
    return pd.DataFrame({
        "time_s": (tl["start"] - t0).dt.total_seconds() if len(tl) else [],
        "status": tl["code"].astype(int) if len(tl) else [],
        "message": tl["code"].map(_CODE_MSG) if len(tl) else [],
        "season": season, "era": ERA_BY_SEASON.get(season, f"{season}_regs"), "source": "openf1",
    }).reset_index(drop=True)


def convert_rcm(aux: dict, season: int) -> pd.DataFrame:
    r = aux["rcm"]
    cols = ["time_utc", "lap", "category", "message", "status", "flag", "scope", "sector", "racing_number",
            "season", "era", "source"]
    if r.empty:
        return pd.DataFrame(columns=cols)
    keep = r["category"].isin(["SafetyCar", "Flag"]) | r["message"].astype(str).str.contains(dl._RCM_KEEP_PATTERN)
    r = r[keep]
    return pd.DataFrame({
        "time_utc": r["date"].dt.strftime("%Y-%m-%d %H:%M:%S"), "lap": r["lap_number"],
        "category": r["category"], "message": r["message"], "status": np.nan, "flag": r["flag"],
        "scope": r["scope"], "sector": r["sector"], "racing_number": r["driver_number"],
        "season": season, "era": ERA_BY_SEASON.get(season, f"{season}_regs"), "source": "openf1",
    }).reset_index(drop=True)[cols]


def convert_pit_stops(laps: pd.DataFrame, aux: dict, season: int) -> pd.DataFrame:
    """dl.pit_stops_from_laps plus OpenF1's stationary time (stop_duration_s)."""
    ps = dl.pit_stops_from_laps(laps)
    pit = aux["pit"]
    ps["stop_duration_s"] = np.nan
    if len(pit) and len(ps):
        num = laps.drop_duplicates("driver").set_index("driver")["driver_number"]
        stop = {(str(d), n): v for d, n, v in zip(pit["driver_number"], pit["lap_number"], pit["stop_duration"])}
        ps["stop_duration_s"] = [stop.get((num.get(d), n), np.nan) for d, n in zip(ps["driver"], ps["in_lap"])]
        ps["stop_duration_s"] = pd.to_numeric(ps["stop_duration_s"], errors="coerce")
    ps["season"] = season
    ps["era"] = ERA_BY_SEASON.get(season, f"{season}_regs")
    ps["source"] = "openf1"
    return ps


def pit_agreement(laps: pd.DataFrame, jolpica_pits: dict, jolpica_results: dict) -> Optional[dict]:
    """
    Independent check of OpenF1 pit_in_lap against Jolpica's pit-stop table
    (driverId, lap). Returns {"jolpica_stops", "matched", "match_rate",
    "openf1_only"} or None if Jolpica has no pit data for the race.
    """
    races = jolpica_pits["MRData"]["RaceTable"]["Races"]
    if not races or not races[0].get("PitStops"):
        return None
    code = {}
    for r in jolpica_results["MRData"]["RaceTable"]["Races"][0]["Results"]:
        code[r["Driver"]["driverId"]] = r["Driver"]["code"]
    jp = {(code.get(p["driverId"]), int(p["lap"])) for p in races[0]["PitStops"]}
    op = set(zip(laps.loc[laps["pit_in_lap"] == 1, "driver"], laps.loc[laps["pit_in_lap"] == 1, "lap_number"]))
    matched = len(jp & op)
    return {"jolpica_stops": len(jp), "matched": matched, "match_rate": round(matched / len(jp), 4),
            "openf1_only": len(op - jp)}


def lap_quality(laps: pd.DataFrame, results: pd.DataFrame, trimmed: int = 0) -> dict:
    """
    Completeness of the OpenF1 lap table, written to race_info so
    downstream fitting can exclude broken races (T-009 found Miami 2025 with
    ~45% of laps missing and Australia 2026 with a collapsed lap counter).
      self_coverage     : rows / sum(per-driver max lap_number). Detects
                          holes without trusting Jolpica's lap counts
                          (Jolpica under-reports laps for disqualified cars).
      jolpica_ratio     : rows / sum(Jolpica laps), informational.
      inner_missing_laps: lap numbers absent inside a driver's span.
      drivers_with_gaps : drivers with any inner gap.
      finishers_no_final_time: classified finishers whose last lap has no
                          lap_time_seconds (= lap counter off by one).
      lap_time_nan_frac : share of rows without lap_time_seconds.
      usable            : self_coverage >= 0.97 and inner gaps <= 1% of rows
                          and finishers_no_final_time <= 10% of finishers.
    """
    exp = float(results["laps"].sum()) if len(results) else float("nan")
    inner, gappy, span = 0, [], 0
    for d, g in laps.groupby("driver"):
        sp = int(g["lap_number"].max())
        span += sp
        n = sp - int(g["lap_number"].min()) + 1 - g["lap_number"].nunique()
        if n:
            inner += n
            gappy.append(d)
    fin = set(results.loc[results["status"].isin(["Finished", "Lapped"]) | results["status"].astype(str).str.startswith("+"), "driver"])
    last = laps.sort_values("lap_number").groupby("driver").tail(1)
    nofin = int(last[last["driver"].isin(fin)]["lap_time_seconds"].isna().sum())
    self_cov = len(laps) / span if span else float("nan")
    return {
        "self_coverage": round(self_cov, 4),
        "jolpica_ratio": round(len(laps) / exp, 4) if exp else None,
        "inner_missing_laps": int(inner), "drivers_with_gaps": sorted(gappy),
        "finishers_no_final_time": nofin,
        "post_race_rows_trimmed": int(trimmed),
        "lap_time_nan_frac": round(float(laps["lap_time_seconds"].isna().mean()), 4),
        "usable": bool(self_cov >= 0.97 and inner <= 0.01 * len(laps) and nofin <= 0.1 * max(len(fin), 1)),
    }


# ---------------------------------------------------------------- build ----
def build_race(meta: dict, raw_dir: str = RAW_DIR) -> dict:
    """meta = one list_races() entry. Fetches (cached) and converts one race."""
    season = meta["season"]
    raw = fetch_session(meta["session_key"], raw_dir)
    event_name = f"{meta['race_name']}"
    cmap = dl.compound_map_for_event(season, event_name)
    jr = fetch_jolpica(season, meta["round"], "results", raw_dir)
    jq = fetch_jolpica(season, meta["round"], "qualifying", raw_dir)
    quali = convert_quali(jq, season)
    results = convert_results(jr, quali, season)
    result_laps = {r.driver: (r.laps, r.status) for r in results.itertuples()}
    laps, aux = convert_laps(raw, season, cmap, result_laps)
    info = {
        "year": season, "season": season, "era": ERA_BY_SEASON.get(season, f"{season}_regs"),
        "gp": meta["label"], "event_name": event_name, "round": meta["round"],
        "circuit": meta["circuit"], "event_date": meta["date"], "event_format": "unknown",
        "total_laps": int(laps["lap_number"].max()) if len(laps) else None,
        "num_laps": int(laps["lap_number"].max()) if len(laps) else None,
        "num_drivers": int(laps["driver"].nunique()),
        "compound_map": cmap,
        "compound_map_source": (dl.COMPOUND_ALLOCATION_SOURCE.get((season, event_name),
                                "T-001 research note §3.2 (team/Pirelli previews)") if cmap else None),
        "openf1_session_key": meta["session_key"], "source": "openf1+jolpica",
        "data_quality": lap_quality(laps, results, aux["trimmed"]),
    }
    jp = fetch_jolpica(season, meta["round"], "pitstops", raw_dir)
    info["data_quality"]["pit_vs_jolpica"] = pit_agreement(laps, jp, jr)
    return {
        "laps": laps, "results": results, "quali": quali, "weather": convert_weather(aux, season),
        "track_status": convert_track_status(aux, season), "rcm": convert_rcm(aux, season),
        "pit_stops": convert_pit_stops(laps, aux, season), "race_info": info,
    }


def write_race(tables: dict, out_dir: str = STAGING_DIR) -> Dict[str, str]:
    info = tables["race_info"]
    written = dl.save_race_tables(tables, out_dir, info["gp"], info["season"])
    q = tables.get("quali")
    if q is not None and len(q):
        path = os.path.join(out_dir, f"{info['gp']}_{info['season']}_quali.csv")
        q.to_csv(path, index=False)
        written["quali"] = path
    return written
