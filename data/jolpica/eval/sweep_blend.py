"""Sweep the baseline blend weight w in p = (1-w)*PL model + w*rank-frequency table.

Run from repo root: .venv/Scripts/python.exe data/jolpica/eval/sweep_blend.py
Model-only and table-only walk-forward predictions are computed once, then
blended linearly (exactly what predictor.predict_race does), so the sweep
costs one set of simulations per (season, mode). Output: blend_sweep.csv/.png.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
import predict_race as PR  # noqa: E402
from app.core import predictor as P  # noqa: E402

WS = np.round(np.arange(0, 1.0001, 0.05), 2)
hist, _, _ = PR.load(True, PR.next_round(True))
res = hist["results"]


def sweep(season, base_cfg):
    last = int(res.query("season == @season")["round"].max())
    rounds = range(2, last + 1)
    rows = []
    for mode in ("prequali", "grid"):
        cfg = P.PredictorConfig(**{**base_cfg, "blend_baseline": 0.0, "n_sims": 4000})
        model = P.walk_forward(hist, season, rounds, f"model_{mode}", cfg)
        base = P.walk_forward(hist, season, rounds, "champ" if mode == "prequali" else "grid", cfg)
        key = ["round", "driver_id"]
        m = model.merge(base[key + ["p_win", "p_podium", "p_top10"]], on=key, suffixes=("", "_b"))
        for w in WS:
            d = m.copy()
            for ev in ("win", "podium", "top10"):
                d[f"p_{ev}"] = np.clip((1 - w) * m[f"p_{ev}"] + w * m[f"p_{ev}_b"], P.PROB_CLIP, 1 - P.PROB_CLIP)
            s = P.score(d)
            rows.append({"season": season, "mode": mode, "w": w, **s})
    return pd.DataFrame(rows)


cfg25 = dict(w2025_level2=0.0, driver_prior_2025=0.0)  # 2025 has no earlier-season priors
cfg26 = {k: v for k, v in P.PredictorConfig().__dict__.items()}
out = pd.concat([sweep(2025, cfg25), sweep(2026, cfg26)], ignore_index=True)
out.to_csv(Path(__file__).with_name("blend_sweep.csv"), index=False)

pd.set_option("display.width", 200)
for (season, mode), g in out.groupby(["season", "mode"]):
    g = g.set_index("w")
    best = g["win_logloss"].idxmin(), g["podium_logloss"].idxmin(), g["winner_nll"].idxmin()
    print(f"\n{season} {mode}: best w  win={best[0]}  podium={best[1]}  winner_nll={best[2]}")
    print(g.loc[[0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0], ["win_logloss", "podium_logloss", "winner_nll"]].round(4).to_string())

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for j, mode in enumerate(("prequali", "grid")):
        for i, met in enumerate(("win_logloss", "podium_logloss")):
            for season, c in ((2025, "tab:gray"), (2026, "tab:red")):
                g = out[(out.season == season) & (out["mode"] == mode)]
                ax[i][j].plot(g["w"], g[met] - g[met].min(), label=str(season), color=c)
            ax[i][j].axvline(0.25, ls=":", color="k")
            ax[i][j].set_title(f"{mode}: {met} above its minimum")
            ax[i][j].legend()
    ax[1][0].set_xlabel("baseline weight w")
    ax[1][1].set_xlabel("baseline weight w")
    fig.tight_layout()
    fig.savefig(Path(__file__).with_name("blend_sweep.png"), dpi=130)
    print("\nsaved plot")
except ImportError:
    print("matplotlib missing, csv only")
