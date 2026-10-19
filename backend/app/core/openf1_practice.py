"""
Practice long-run pace from OpenF1 (https://api.openf1.org/v1, free, no
auth) as an optional predictor feature (T-011, 2026-10-19).

Why: the orchestrator found OpenF1 still serves 2026 practice laps and
stints while FastF1 live timing is 401 (team/BOARD.md, DATA ACCESS
INCIDENT). Practice long runs are the only pre-quali information about
*this weekend's* race pace. T-010 4.4 found no published number on its
value, so it is only switched on if walk-forward says it helps
(PredictorConfig.use_fp; see the T-011 handoff for the result).

Long-run definition (crude on purpose, fuel loads and engine modes are
unknown):
- per driver stint (OpenF1 /stints) drop the pit-out lap and the first
  lap of the stint, laps without a lap_duration, and laps slower than
  1.03 x the stint median (cool-down / traffic laps);
- keep stints with >= MIN_RUN_LAPS remaining laps;
- stint pace = median lap; gap = stint pace - best stint pace *on the same
  compound in the same session* (a crude compound correction);
- driver gap for the weekend = lap-weighted mean over his long-run stints
  across all practice sessions of the meeting.
The predictor uses a team-level gap (mean of the team's drivers), because
per-driver run plans differ more than per-team ones (T-010 4.4).

Cache: data/openf1_predictor/raw/ (separate from data/openf1/, which T-009
owns). Only non-empty responses are cached; only the fields used are kept
(the laps endpoint carries ~250 KB of mini-sector codes per session).
Requests are spaced MIN_REQUEST_INTERVAL_S apart (OpenF1's free tier
allows a few per second).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://api.openf1.org/v1"
REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = REPO_ROOT / "data" / "openf1_predictor" / "raw"
MIN_REQUEST_INTERVAL_S = 1.0
SESSIONS_TTL_S = 3 * 3600
MIN_RUN_LAPS = 5
SLOW_LAP_FACTOR = 1.03

LAP_FIELDS = ["driver_number", "lap_number", "lap_duration", "is_pit_out_lap"]
_last_t = 0.0


def _get(endpoint: str, params: dict, cache_name: str, ttl_s: float | None = None,
         keep: list[str] | None = None, offline: bool = False) -> list[dict]:
    global _last_t
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cp = RAW_DIR / f"{cache_name}.json"
    if cp.exists() and (offline or ttl_s is None or time.time() - cp.stat().st_mtime < ttl_s):
        return json.loads(cp.read_text(encoding="utf-8"))
    if offline:
        return []
    for attempt in range(4):
        wait = MIN_REQUEST_INTERVAL_S - (time.monotonic() - _last_t)
        if wait > 0:
            time.sleep(wait)
        _last_t = time.monotonic()
        r = requests.get(f"{BASE_URL}/{endpoint}", params=params, timeout=60)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 404:  # OpenF1 answers 404 for "no rows"
            return []
        r.raise_for_status()
        data = r.json()
        if keep:
            data = [{k: row.get(k) for k in keep} for row in data]
        if data:
            cp.write_text(json.dumps(data), encoding="utf-8")
        return data
    raise RuntimeError(f"OpenF1 request failed: {endpoint} {params}")


def practice_sessions(year: int, offline: bool = False) -> pd.DataFrame:
    rows = _get("sessions", {"year": year}, f"sessions_{year}", SESSIONS_TTL_S, offline=offline)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date_start"], utc=True)
    return df


def _session_tables(session_key: int, offline: bool):
    laps = pd.DataFrame(_get("laps", {"session_key": session_key}, f"laps_{session_key}",
                             keep=LAP_FIELDS, offline=offline))
    stints = pd.DataFrame(_get("stints", {"session_key": session_key}, f"stints_{session_key}",
                               offline=offline))
    drivers = pd.DataFrame(_get("drivers", {"session_key": session_key}, f"drivers_{session_key}",
                                keep=["driver_number", "name_acronym", "team_name"], offline=offline))
    return laps, stints, drivers


def long_run_stints(laps: pd.DataFrame, stints: pd.DataFrame) -> pd.DataFrame:
    """One row per qualifying long-run stint: driver_number, compound, n, pace."""
    if laps.empty or stints.empty:
        return pd.DataFrame(columns=["driver_number", "compound", "n", "pace"])
    out = []
    for _, s in stints.dropna(subset=["lap_start", "lap_end"]).iterrows():
        l = laps[(laps["driver_number"] == s["driver_number"])
                 & (laps["lap_number"] > s["lap_start"])  # drops the stint's first lap
                 & (laps["lap_number"] <= s["lap_end"])]
        l = l[(l["is_pit_out_lap"] != True) & l["lap_duration"].notna()]  # noqa: E712
        if len(l) < MIN_RUN_LAPS:
            continue
        t = l["lap_duration"].astype(float)
        t = t[t <= SLOW_LAP_FACTOR * t.median()]
        if len(t) < MIN_RUN_LAPS:
            continue
        out.append({"driver_number": s["driver_number"], "compound": s.get("compound"),
                    "n": len(t), "pace": float(t.median())})
    return pd.DataFrame(out, columns=["driver_number", "compound", "n", "pace"])


def weekend_long_run_gaps(year: int, offline: bool = False) -> pd.DataFrame:
    """
    Per (race date, driver code, team) long-run gap in seconds (>= 0, lower
    is faster) over all practice sessions of each meeting held so far.
    `race_date` (YYYY-MM-DD of the Race session) is the join key to the
    Jolpica schedule.
    """
    ses = practice_sessions(year, offline)
    if ses.empty:
        return pd.DataFrame(columns=["race_date", "code", "team_name", "fp_gap", "fp_laps"])
    races = ses[ses["session_name"] == "Race"][["meeting_key", "date"]]
    race_date = dict(zip(races["meeting_key"], races["date"].dt.strftime("%Y-%m-%d")))
    # only sessions that ended >= 30 min ago: a session fetched while live
    # would be cached half-complete forever
    now = pd.Timestamp.now(tz="UTC")
    ended = pd.to_datetime(ses["date_end"], utc=True) < now - pd.Timedelta(minutes=30)
    fps = ses[(ses["session_type"] == "Practice") & ended]
    if "is_cancelled" in fps:
        fps = fps[fps["is_cancelled"] != True]  # noqa: E712
    rows = []
    for _, s in fps.iterrows():
        laps, stints, drivers = _session_tables(int(s["session_key"]), offline)
        lr = long_run_stints(laps, stints)
        if lr.empty or drivers.empty:
            continue
        lr["gap"] = lr["pace"] - lr.groupby("compound")["pace"].transform("min")
        lr = lr.merge(drivers.drop_duplicates("driver_number"), on="driver_number", how="left")
        lr["race_date"] = race_date.get(s["meeting_key"])
        lr["session_name"] = s["session_name"]
        rows.append(lr)
    if not rows:
        return pd.DataFrame(columns=["race_date", "code", "team_name", "fp_gap", "fp_laps"])
    df = pd.concat(rows, ignore_index=True)
    g = df.groupby(["race_date", "name_acronym", "team_name"]).apply(
        lambda x: pd.Series({"fp_gap": np.average(x["gap"], weights=x["n"]), "fp_laps": x["n"].sum()}),
        include_groups=False,
    ).reset_index()
    return g.rename(columns={"name_acronym": "code"})
