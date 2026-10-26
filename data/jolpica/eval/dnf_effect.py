"""T-045: effect of the DNF changes on win/podium/top10 log-loss (grid + prequali), 2025 and 2026."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "backend"))
import pandas as pd
import predict_race as PR
from app.core import predictor as P

hist, _, _ = PR.load(True, 19)
base = {**PR.CONFIG.__dict__, "n_sims": 20000}
VAR = {"before (m 10/20, no grid factor)": dict(dnf_team_m=10.0, dnf_driver_m=20.0, dnf_grid_power=0.0),
       "shrink only (20/60)": dict(dnf_grid_power=0.0),
       "after (20/60 + grid factor .5)": dict()}
rows = []
for name, v in VAR.items():
    cfg = P.PredictorConfig(**{**base, **v})
    for season, rr in ((2025, range(2, 25)), (2026, range(2, 19))):
        for mode in ("grid", "prequali"):
            s = P.score(P.walk_forward(hist, season, rr, f"model_{mode}", cfg))
            rows.append(dict(variant=name, season=season, mode=mode, **s))
t = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(t.drop(columns=["win_brier", "podium_brier", "top10_brier"]).round(4).to_string(index=False))
