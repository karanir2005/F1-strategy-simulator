"""T-045: driver-level walk-forward tables (grid mode) for 2025 and 2026 with every ranking
statistic, used by the ranking-statistic comparison and the inversion audit.

    python data/jolpica/eval/rank_audit.py            # writes eval/rank_audit_drivers.csv
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
import numpy as np
import pandas as pd

import predict_race as PR
from app.core import predictor as P


def driver_table(hist, season, rounds, cfg):
    stores = P._stores_for(hist, cfg, season)
    res = hist["results"]
    rows = []
    for rnd in rounds:
        p = P.predict_race(hist, season, rnd, "grid", cfg, stores=stores)
        a = res[(res["season"] == season) & (res["round"] == rnd)].set_index("driver_id")
        p = p.copy()
        p["actual_position"] = p["driver_id"].map(a["position"])
        p["actual_classified"] = p["driver_id"].map(a["classified"])
        p["actual_status"] = p["driver_id"].map(a["status"])
        rows.append(p)
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    hist, _, _ = PR.load(True, 19)
    cfg = P.PredictorConfig(**{**PR.CONFIG.__dict__, "n_sims": 20000})
    out = []
    for season, rounds in ((2025, range(2, 25)), (2026, range(1, 19))):
        out.append(driver_table(hist, season, rounds, cfg))
    d = pd.concat(out, ignore_index=True)
    d.to_csv(Path(__file__).parent / "rank_audit_drivers.csv", index=False)
    print(len(d))
