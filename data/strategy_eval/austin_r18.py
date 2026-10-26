"""Austin R18 (2026): what the optimizer recommends (before/after Change A, shipped vs clean caps) vs the actual top 10."""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend")); sys.path.insert(0, str(Path(__file__).parent))
from app.core import tire_model as tm
from app.core.optimizer import optimize
import eval_strategy as ev
races, _ = ev.load_races()
r = races[ev.AUSTIN26]
print("ACTUAL top 10 (R18, 56 laps):")
for i, d in enumerate(r["order"][:10], 1):
    s = r["strat"][d]; print(f"  P{i:<2} {d}  {ev.fmt(s):<26} stops {len(s)-1}  pits {ev.pit_laps(s)}")
T = ev.TRACKS["austin"]
laps26 = tm.load_all_laps(eras=("2026_regs",))
out = {}
for mode in ("joint", "pace_ordered"):
    model = tm.build_tire_model(laps=laps26, eras=("2026_regs",), offset_mode=mode)
    p90, mx = ev.stint_caps(races, "2026_regs", exclude="none")
    for capname, (maxc, typ) in {"shipped_track_caps(2025 Austin)": (T["max_stint_laps"], T.get("typical_stint_laps")), "clean_pooled_2026_caps": (mx, p90)}.items():
        res = optimize(model, 56, T["pit_loss_s"], T["base_laptime_s"], maxc, typical_stint_laps=typ, max_stops=3, n_sims=2000, top_n=5)
        print(f"\n[{mode}] offsets {({k: round(v,3) for k,v in model['_compound_offsets'].items()})} caps {res.effective_caps} ({capname})")
        for p in res.plans:
            print(f"  #{p['rank']} {ev.fmt([(s['compound'], s['laps']) for s in p['stints']]):<22} pits {p['pit_laps']}  +{p['delta_to_best_mean_s']:.1f}s")
        out[f"{mode}|{capname}"] = [ev.fmt([(s['compound'], s['laps']) for s in p['stints']]) for p in res.plans]
