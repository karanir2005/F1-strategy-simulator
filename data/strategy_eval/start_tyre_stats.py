"""
Start-tyre and compound-durability statistics from real top-10 strategies (T-047).

    .venv/Scripts/python.exe data/strategy_eval/start_tyre_stats.py

Writes data/strategy_eval/start_tyre_stats.json. Evidence for optimizer.START_RANK (softer compound first among
exact model ties) and for the per-compound stint-length spread. Dry, red-flag-free races only (eval_strategy.load_races).
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_strategy as e  # noqa: E402

rank = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}
races, _ = e.load_races()
out = {}
for era in ("2026_regs", "2025_regs"):
    first, lens = Counter(), {c: [] for c in rank}
    for r in races.values():
        if r["era"] != era:
            continue
        for d in r["order"][:10]:
            s = [x for x in r["strat"].get(d, []) if x[0] in rank]
            for i, (c, n) in enumerate(s[:-1]):
                lens[c].append(n)  # closed stints only (the last one is cut by the flag)
            cs = [c for c, _ in s]
            if len(s) == 2 and cs[0] != cs[1]:
                first["softer_first" if rank[cs[0]] < rank[cs[1]] else "harder_first"] += 1
    out[era] = dict(
        one_stop_two_compound_first_stint=dict(first),
        softer_first_share=round(first["softer_first"] / max(1, sum(first.values())), 3),
        closed_stint_laps={c: dict(n=len(v), median=float(np.median(v)), p90=float(np.percentile(v, 90))) for c, v in lens.items() if v},
    )
Path(__file__).with_name("start_tyre_stats.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
