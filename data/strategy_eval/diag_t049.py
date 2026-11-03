"""T-049 step 1 diagnostics (T-047): is the 0.037 s/lap wear slope a truncation / fuel artefact?
  a) wear slope on laps with age <= K vs the full range (free joint fit, 2026 and 2025 eras)
  b) wear slope with the fuel slope FIXED at (fitted +/- 0.01 s/lap), same global value for all races
  c) cliff_age / max_reliable_age of the shipped model; laps beyond the pooled p50/p75/p90 stint length
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend")); sys.path.insert(0, str(Path(__file__).parent))
from app.core import tire_model as tm
import eval_strategy as ev

races, _ = ev.load_races()
C = ("SOFT", "MEDIUM", "HARD")

def wear_fixed_fuel(clean, fuel, offsets):
    y = clean["lap_time_seconds"] - clean["compound"].map(offsets).astype(float) - fuel * clean["lap_number"]
    g = [clean["race_file"], clean["driver"]]
    X = pd.DataFrame({c: (clean["compound"] == c) * clean["tire_age"].astype(float) for c in C}, index=clean.index)
    Xw = X - X.groupby(g).transform("mean"); yw = y - y.groupby(g).transform("mean")
    b, *_ = np.linalg.lstsq(Xw.values, yw.values, rcond=None)
    return dict(zip(C, np.round(b, 4)))

for era in ("2026_regs", "2025_regs"):
    laps = tm.assign_stint_ids(tm.load_all_laps(eras=(era,)))
    clean = tm.clean_laps(laps)
    free = tm.fit_joint_model(clean)
    print(f"\n== {era}: {len(clean)} clean laps; free fit wear", {c: round(free['wear'][c][0], 4) for c in C},
          "offsets", {c: round(v, 3) for c, v in free['offsets'].items()}, "fuel", round(free['fuel_coef'], 4))
    for K in (12, 18, 25, 99):
        sub = clean[clean["tire_age"] <= K]
        f = tm.fit_joint_model(sub)
        print(f"  age<={K:<3} n={len(sub):<6}", {c: round(f['wear'][c][0], 4) for c in C})
    off = {c: free["offsets"][c] for c in C}
    for d in (-0.01, 0.0, 0.01):
        print(f"  fuel fixed {free['fuel_coef']+d:+.4f} (d={d:+.2f}), offsets from free fit:", wear_fixed_fuel(clean, free["fuel_coef"] + d, off))
    m = tm.build_tire_model(laps=laps, eras=(era,))
    print("  shipped-form model: cliff_age", {c: m[c]["cliff_age"] for c in C}, "max_reliable_age", {c: m[c]["max_reliable_age"] for c in C})
    for pct in (50, 75, 90):
        L, _ = ev.stint_caps(races, era, exclude="__none__", pct=pct)
        n = {c: int(((clean["compound"] == c) & (clean["tire_age"] > L.get(c, 99))).sum()) for c in C}
        tot = {c: int((clean["compound"] == c).sum()) for c in C}
        print(f"  laps beyond pooled p{pct} stint length {L}:", n, "of", tot)
