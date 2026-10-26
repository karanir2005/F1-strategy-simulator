"""Diagnostic (T-043): stint-length distributions per compound and cohort wear curves, 2026 vs 2025."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.core import tire_model as tm
sys.path.insert(0, str(Path(__file__).parent))
import eval_strategy as ev

races, _ = ev.load_races()
for era in ("2026_regs", "2025_regs"):
    print("==", era)
    for c in ("SOFT", "MEDIUM", "HARD"):
        v = [n for k, r in races.items() if r["era"] == era for d, s in r["strat"].items()
             for i, (cc, n) in enumerate(s) if cc == c and n >= 5]
        top = [n for k, r in races.items() if r["era"] == era for d in r["order"][:10] if d in r["strat"]
               for (cc, n) in r["strat"][d] if cc == c and n >= 5]
        print(c, "all n=%d p50 %.0f p75 %.0f p90 %.0f max %d | top10 n=%d p50 %.0f p90 %.0f" % (
            len(v), *np.percentile(v, [50, 75, 90]), max(v), len(top), *np.percentile(top, [50, 90])))
    laps = tm.load_all_laps(eras=(era,))
    m = tm.build_tire_model(laps=laps, eras=(era,))
    lp = tm.assign_stint_ids(laps)
    clean = tm.clean_laps(lp)
    fuel = clean["race_file"].map(lambda f: m["_fuel_coef_by_race"][Path(f).name])
    y = clean["lap_time_seconds"] - fuel * clean["lap_number"]
    clean = clean.assign(y=y)
    # cohort: stints with >= 25 clean laps; level = mean y of first 5 clean laps of the stint
    rows = []
    for (rf, d, sid, c), g in clean.groupby(["race_file", "driver", "stint_id", "compound"]):
        g = g.sort_values("tire_age")
        if len(g) < 18: continue
        base = g["y"].iloc[:4].mean(); a0 = g["tire_age"].iloc[:4].mean()
        for a, yy in zip(g["tire_age"], g["y"]):
            rows.append((c, int(a), yy - base, a0))
    df = pd.DataFrame(rows, columns=["c", "age", "dy", "a0"])
    for c in ("SOFT", "MEDIUM", "HARD"):
        s = df[df.c == c]
        out = []
        for lo in range(5, 45, 5):
            b = s[(s.age >= lo) & (s.age < lo + 5)]
            out.append("%d-%d: %+.2f (n=%d)" % (lo, lo + 4, b.dy.mean(), len(b)) if len(b) >= 15 else "%d-%d: -" % (lo, lo + 4))
        print(c, " | ".join(out))
