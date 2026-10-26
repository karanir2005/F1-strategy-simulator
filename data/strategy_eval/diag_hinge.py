"""Diagnostic (T-043): is there a measurable extra wear slope beyond the typical stint length? OLS of the joint-fit
residual on hinge = max(0, age - L_c), L_c = pooled p<pct> observed stint length of compound c."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend")); sys.path.insert(0, str(Path(__file__).parent))
from app.core import tire_model as tm
import eval_strategy as ev
races, _ = ev.load_races()
for era in ("2026_regs", "2025_regs"):
    laps = tm.load_all_laps(eras=(era,))
    clean = tm.clean_laps(tm.assign_stint_ids(laps))
    fit = tm.fit_joint_model(clean, constrain=True)
    res = fit["residual"]
    for pct in (50, 75, 90):
        L, _ = ev.stint_caps(races, era, exclude="__none__", pct=pct)
        h = clean.apply(lambda r: max(0.0, r["tire_age"] - L.get(r["compound"], 99)), axis=1)
        g = [clean["race_file"], clean["driver"]]
        hw = h - h.groupby(g).transform("mean"); rw = res - res.groupby(g).transform("mean")
        out = {}
        for c in ("SOFT", "MEDIUM", "HARD"):
            m = (clean["compound"] == c).values
            x = hw[m].values; y = rw[m].values
            out[c] = (round(float(x @ y / (x @ x)), 3), int((h[m] > 0).sum()), L.get(c))
        x = hw.values; y = rw.values
        print(era, "p%d" % pct, "pooled k = %.3f" % float(x @ y / (x @ x)), out)
