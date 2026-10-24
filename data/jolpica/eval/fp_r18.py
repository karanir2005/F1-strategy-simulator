"""T-041: R18 (Austin) pre-quali top 5 with and without FP (FP1/FP2 data available so far)."""
import sys
from dataclasses import replace
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fp_experiment as E
PR, P = E.PR, E.P
hist, _, sched = PR.load(True, 18)
s = E.sess_tables(); var = E.variants(s, sched[["season", "round", "date"]])
mode, ent, grid, src, gs, hq = PR.resolve(hist, 18, "prequali", None)
cfg = P.PredictorConfig(n_sims=40000)
print("R18 FP rows (code count):", {n: int((v["round"] == 18).sum()) for n, v in var.items()})
sess = s[s["race_date"] == "2026-10-25"]; print(sess.groupby("session")[["best_lap", "lr_gap"]].count())
out = {}
t = P.predict_race(hist, 2026, 18, "prequali", cfg, entry_list=ent, grid=grid); out["no FP"] = t
for n in ("V0 all long-run", "C1 fastest lap"):
    h = dict(hist); h["fp"] = var[n]
    t = P.predict_race(h, 2026, 18, "prequali", replace(cfg, use_fp=True), entry_list=ent, grid=grid)
    out[n] = t; print(n, "b_fp =", round(t.attrs["model"].b_fp, 3))
for n, t in out.items():
    print("\n", n); print(t[["code", "p_win", "p_podium", "p_top10"]].head(6).round(3).to_string(index=False))
