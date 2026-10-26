"""Second sensitivity grid (T-043) on top of Change A (pace_ordered): caps / stint-life cliff. Not shipped, kept as evidence."""
import sys, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import eval_strategy as ev
import pandas as pd
base = {"model_kwargs": {"offset_mode": "pace_ordered"}, "quiet": True}
cfgs = {"A": {}, "A_cap75": {"cap_pct": 75}, "A_cap60": {"cap_pct": 60}}
for lp, k in itertools.product((50, 75), (0.1, 0.2, 0.4)):
    cfgs[f"A_life{lp}_k{k}"] = {"life_pct": lp, "cliff_k": k}
cfgs["A_cap75_life75_k0.2"] = {"cap_pct": 75, "life_pct": 75, "cliff_k": 0.2}
rows = []
for name, c in cfgs.items():
    d = ev.run(dict(base, **c), 200)
    s = ev.summarise(d)
    for era in ("all", "2026_regs", "2025_regs"):
        b = s[era]
        rows.append(dict(cfg=name, era=era, **{k: b[k] for k in ("stops_match_modal", "stops_match_modal_green", "set_match_modal", "modal_in_opt_top3", "median_winner_gap_s", "median_pit_err_green_laps", "mean_opt_stops", "soft_share_opt")}))
pd.DataFrame(rows).to_csv(ev.OUT_DIR / "grid2_results.csv", index=False)
print(pd.DataFrame(rows).to_string())
