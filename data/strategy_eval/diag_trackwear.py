import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend")); sys.path.insert(0, str(Path(__file__).parent))
from app.core import tire_model as tm
import eval_strategy as ev
m26 = tm.build_tire_model(); m25 = tm.build_tire_model(laps=tm.load_all_laps(eras=(tm.ERA_2025,)), eras=(tm.ERA_2025,))
races, _ = ev.load_races()
rows = []
for t, w26 in m26["_track_wear"].items():
    w25 = m25["_track_wear"].get(t)
    if not w25: continue
    g26 = np.mean([w26[c]["age_coef"] for c in ("MEDIUM","HARD")]); g25 = np.mean([w25[c]["age_coef"] for c in ("MEDIUM","HARD")])
    r = {k: v for k, v in races.items() if tm.track_id_of_race(k) == t}
    st = {k.split("_")[1]: np.mean([len(s) - 1 for d, s in v["strat"].items() if d in v["order"][:10]]) for k, v in r.items()}
    rows.append((t, g26, g25, st.get("2026"), st.get("2025")))
d = pd.DataFrame(rows, columns=["track", "w26", "w25", "stops26", "stops25"])
print(d.round(3).to_string()); print(d.drop(columns="track").corr().round(2))
