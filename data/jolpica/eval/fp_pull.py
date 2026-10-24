"""T-041: per-session practice tables (2025, 2026) from OpenF1 for the FP experiment.
Output data/jolpica/eval/fp_sessions_{year}.csv, one row per (race_date, session, driver):
best lap (s), long-run pace/gap (lap-weighted over the session's long-run stints), wet flag.
Uses the T-011 cache (data/openf1_predictor/raw); only live-fetches what is missing."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "backend"))
from app.core import openf1_practice as F

def build(year, offline=False):
    ses = F.practice_sessions(year, offline)
    races = ses[ses["session_name"] == "Race"][["meeting_key", "date"]]
    rd = dict(zip(races["meeting_key"], races["date"].dt.strftime("%Y-%m-%d")))
    now = pd.Timestamp.now(tz="UTC")
    ended = pd.to_datetime(ses["date_end"], utc=True) < now - pd.Timedelta(minutes=30)
    fps = ses[(ses["session_type"] == "Practice") & ended]
    if "is_cancelled" in fps:
        fps = fps[fps["is_cancelled"] != True]
    rows = []
    for _, s in fps.iterrows():
        laps, stints, drivers = F._session_tables(int(s["session_key"]), offline)
        if laps.empty or drivers.empty:
            continue
        wet = bool(len(stints) and stints["compound"].isin(["INTERMEDIATE", "WET"]).any())
        l = laps[(laps["is_pit_out_lap"] != True) & laps["lap_duration"].notna()]
        best = l.groupby("driver_number")["lap_duration"].min().rename("best_lap")
        lr = F.long_run_stints(laps, stints)
        if len(lr):
            lr["gap"] = lr["pace"] - lr.groupby("compound")["pace"].transform("min")
            g = lr.groupby("driver_number").apply(lambda x: pd.Series({"lr_gap": np.average(x["gap"], weights=x["n"]), "lr_n": x["n"].sum()}), include_groups=False)
        else:
            g = pd.DataFrame(columns=["lr_gap", "lr_n"])
        d = drivers.drop_duplicates("driver_number").set_index("driver_number")
        t = d.join(best).join(g).reset_index()
        t["race_date"] = rd.get(s["meeting_key"]); t["session"] = s["session_name"]
        t["session_key"] = s["session_key"]; t["wet"] = wet; t["start"] = s["date_start"]
        rows.append(t)
    return pd.concat(rows, ignore_index=True).rename(columns={"name_acronym": "code"})

if __name__ == "__main__":
    for y in (2025, 2026):
        df = build(y)
        df.to_csv(Path(__file__).with_name(f"fp_sessions_{y}.csv"), index=False)
        print(y, df["race_date"].nunique(), "weekends", len(df), "rows", df["wet"].sum(), "wet rows")
