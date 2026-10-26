"""Phase-2 sensitivity grid (T-043): each config is one eval_strategy.run(); prints agreement metrics."""
import sys, json, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import eval_strategy as ev
import pandas as pd

configs = {"baseline": {}}
for pct in (50, 60, 75):
    configs[f"cap_p{pct}"] = {"cap_pct": pct}
configs["soft0"] = {"offsets": {"SOFT": 0.0}}
configs["soft0_cap75"] = {"offsets": {"SOFT": 0.0}, "cap_pct": 75}
for lp, k in itertools.product((50, 75), (0.05, 0.1, 0.2)):
    configs[f"life{lp}_k{k}"] = {"life_pct": lp, "cliff_k": k}
    configs[f"soft0_life{lp}_k{k}"] = {"offsets": {"SOFT": 0.0}, "life_pct": lp, "cliff_k": k}
rows = []
for name, cfg in configs.items():
    d = ev.run(dict(cfg, quiet=True), 200)
    d.to_csv(ev.OUT_DIR / "grid" / f"{name}.csv", index=False) if (ev.OUT_DIR / "grid").exists() else None
    s = ev.summarise(d)
    for era in ("all", "2026_regs", "2025_regs"):
        b = s[era]
        rows.append(dict(cfg=name, era=era, **{k: b[k] for k in ("stops_match_modal", "seq_match_modal", "set_match_modal", "modal_in_opt_top3", "median_winner_gap_s", "median_pit_err_laps", "mean_opt_stops", "soft_share_opt", "med_share_opt", "hard_share_opt")}))
    print(name, s["all"]["stops_match_modal"], s["all"]["set_match_modal"], s["all"]["modal_in_opt_top3"], s["2026_regs"]["stops_match_modal"], s["2026_regs"]["set_match_modal"], s["2026_regs"]["soft_share_opt"], flush=True)
pd.DataFrame(rows).to_csv(ev.OUT_DIR / "grid_results.csv", index=False)
