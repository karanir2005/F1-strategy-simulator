"""Diagnostic (T-043): effective time lost to a stop (in-lap + out-lap minus the two neighbouring normal laps) vs the
pit-lane duration the simulator is charged (track_params pit_loss_s)."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend")); sys.path.insert(0, str(Path(__file__).parent))
from app.core import tire_model as tm
import eval_strategy as ev

def stops(laps, key):
    out = []
    laps = laps.sort_values(["driver", "lap_number"])
    for d, g in laps.groupby("driver"):
        g = g.set_index("lap_number")
        for lap in g.index[g["pit_in_lap"].astype(bool)]:
            o = lap + 1
            if o not in g.index or not bool(g.loc[o, "pit_out_lap"]): continue
            pv, nx = lap - 1, o + 1
            if pv not in g.index or nx not in g.index: continue
            ok = lambda l: bool(g.loc[l, "is_green_flag"]) and not g.loc[l, "pit_in_lap"] and not g.loc[l, "pit_out_lap"]
            if not (ok(pv) and ok(nx)): continue
            if g.loc[[lap, o], "is_neutralised"].astype(bool).any(): continue
            if not bool(g.loc[lap, "is_tyre_change"]) and not bool(g.loc[o, "is_tyre_change"]): continue
            t = g.loc[[pv, lap, o, nx], "lap_time_seconds"].values
            if np.isnan(t).any(): continue
            out.append(dict(race=key, driver=d, lap=lap, loss=t[1] + t[2] - t[0] - t[3], pit_lane=g.loc[o, "pit_lane_s"] if "pit_lane_s" in g else np.nan,
                            in_loss=t[1] - t[0], out_loss=t[2] - t[3]))
    return out
rows = []
races, _ = ev.load_races()
for k, r in races.items():
    if r["laps"] is not None:
        rows += stops(r["laps"], k)
d = pd.DataFrame(rows)
d["pit_lane"] = pd.to_numeric(d["pit_lane"], errors="coerce")
d = d[(d.loss > 5) & (d.loss < 60)]
print(len(d), "stops; loss median %.1f mean %.1f | pit_lane median %.1f" % (d.loss.median(), d.loss.mean(), d.pit_lane.median()))
print(d.groupby(d.race.str.split("_").str[1])[["loss", "pit_lane", "in_loss", "out_loss"]].median().round(1))
g = d.groupby("race").agg(n=("loss", "size"), loss=("loss", "median"), pit_lane=("pit_lane", "median")); g["tp"] = [ev.TRACKS.get(tm.track_id_of_race(k), {}).get("pit_loss_s") for k in g.index]
g["diff"] = g.loss - g.tp
print(g.round(1).to_string()); print("median diff (loss - track pit_loss_s) %.1f; corr %.2f" % (g["diff"].median(), g[["loss", "tp"]].corr().iloc[0, 1]))
d.to_csv(ROOT / "data" / "strategy_eval" / "stop_time_loss.csv", index=False)
