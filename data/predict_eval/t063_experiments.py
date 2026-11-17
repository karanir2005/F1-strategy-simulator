"""
T-063 (2026-11-16): try to improve predictor calibration under strict walk-forward.
Run from repo root:  .venv/Scripts\python data\predict_eval\t063_experiments.py [exp ...]
In-memory only (never calls --by-race); writes data/predict_eval/out/t063_*.csv.
Experiments: A (in-model knobs), B (pooled shrunk Platt), D (outsider-bin audit).
Judged on held-out per-driver Brier / log loss for win, podium, top10, winner NLL, ECE, hit rates,
paired race bootstrap of (variant - default) over races.
"""
import sys
from dataclasses import replace
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend")); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
from scipy.optimize import minimize
import predict_race as PR
from app.core import predictor as P
import calibration_check as C

OUT = Path(__file__).parent / "out"
EV = ("win", "podium", "top10")


def metrics(df):
    r = {}
    for ev in EV:
        p, y = df[f"p_{ev}"].to_numpy(), df[f"y_{ev}"].to_numpy()
        r[f"brier_{ev}"], r[f"ll_{ev}"], r[f"ece_{ev}"] = C.brier(p, y), C.logloss(p, y), C.ece(p, y, ev)
    w = df[df.y_win == 1]
    r["winner_nll"] = float(-np.log(w.p_win).mean())
    g = df.groupby("round")
    r["win_hit"] = int(sum(x.loc[x.p_win.idxmax(), "y_win"] for _, x in g))
    r["pod_hit"] = int(sum(x.nlargest(3, "p_podium").y_podium.sum() for _, x in g))
    r["n_races"] = df["round"].nunique()
    return r


def boot(df, base, ev, kind, n=3000, seed=0):
    """paired race bootstrap of mean(loss variant) - mean(loss base); aligned rows."""
    key = ["round", "driver_id"]  # walk_forward rows are sorted by p_win, so align explicitly
    df = df.sort_values(key).reset_index(drop=True); base = base.sort_values(key).reset_index(drop=True)
    p, q, y = df[f"p_{ev}"].to_numpy(), base[f"p_{ev}"].to_numpy(), df[f"y_{ev}"].to_numpy()
    if kind == "brier":
        a, b = (p - y) ** 2, (q - y) ** 2
    else:
        pc, qc = np.clip(p, 1e-4, 1 - 1e-4), np.clip(q, 1e-4, 1 - 1e-4)
        a, b = -(y * np.log(pc) + (1 - y) * np.log(1 - pc)), -(y * np.log(qc) + (1 - y) * np.log(1 - qc))
    d = pd.DataFrame({"r": df["round"].to_numpy(), "d": a - b}).groupby("r")["d"].agg(["sum", "count"]).to_numpy()
    rng = np.random.default_rng(seed)
    out = [d[s, 0].sum() / d[s, 1].sum() for s in (rng.choice(len(d), len(d)) for _ in range(n))]
    return float(d[:, 0].sum() / d[:, 1].sum()), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def run(hist, season, cfg):
    last = int(hist["results"].query("season==@season")["round"].max())
    return P.walk_forward(hist, season, range(1, last + 1), "model_grid", cfg)


def main():
    exps = sys.argv[1:] or ["A", "B", "D"]
    hist, _, _ = PR.load(True, 99)
    base_cfg = P.PredictorConfig(**{**PR.CONFIG.__dict__})
    rows = []
    cache = {}
    if "Arecompute" in exps:
        cache = pd.read_pickle(OUT / "t063_cache.pkl")
        for (season, name), df in cache.items():
            m = metrics(df); m.update(season=season, variant=name)
            if name != "default":
                for ev in EV:
                    for kind in ("brier", "ll"):
                        d, lo, hi = boot(df, cache[(season, "default")], ev, kind)
                        m[f"d{kind}_{ev}"] = f"{d:+.4f}[{lo:+.4f},{hi:+.4f}]"
            rows.append(m)
        pd.DataFrame(rows).to_csv(OUT / "t063_A.csv", index=False); cache = {}
    if "A" in exps:
        variants = {"default": {}}
        for s in (0.85, 1.15, 1.3, 1.5): variants[f"scale={s}"] = {"utility_scale": s}
        for b in (0.0, 0.1, 0.4): variants[f"blend={b}"] = {"blend_baseline": b}
        for b in (0.1, 0.25): variants[f"blend_win={b}"] = {"blend_win_grid": b}
        for g in (0.0, 2.0, 6.0): variants[f"gamma={g}"] = {"stick_gamma": g}
        for k in (5, 10): variants[f"topk2={k}"] = {"topk_level2": k}
        variants["w2025=0"] = {"w2025_level2": 0.0}
        variants["scale1.3+blend0.1"] = {"utility_scale": 1.3, "blend_baseline": 0.1}
        for season in (2026, 2025):
            for name, kw in variants.items():
                cfg = replace(base_cfg, **kw)
                df = run(hist, season, cfg)
                cache[(season, name)] = df
                m = metrics(df); m.update(season=season, variant=name)
                if name != "default":
                    b0 = cache[(season, "default")]
                    for ev in EV:
                        for kind in ("brier", "ll"):
                            d, lo, hi = boot(df, b0, ev, kind)
                            m[f"d{kind}_{ev}"] = f"{d:+.4f}[{lo:+.4f},{hi:+.4f}]"
                rows.append(m); print(season, name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if k.startswith(("ll_", "winner", "win_hit", "pod_hit"))}, flush=True)
        pd.DataFrame(rows).to_csv(OUT / "t063_A.csv", index=False)
    pd.to_pickle(cache, OUT / "t063_cache.pkl") if cache else None



# ---------------------------------------------------------------- B / C: walk-forward recalibrators
def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))


def _sig(z):
    return 1 / (1 + np.exp(-z))


def fit_platt(p, y, lam, slope_only=False):
    """argmin_{a,b} sum logloss(sigmoid(a*logit p + b)) + lam*((a-1)^2 + b^2)  (prior toward identity)."""
    z = _logit(p)
    def f(x):
        a, b = x[0], (0.0 if slope_only else x[1])
        q = np.clip(_sig(a * z + b), 1e-6, 1 - 1e-6)
        return -(y * np.log(q) + (1 - y) * np.log(1 - q)).sum() + lam * ((a - 1) ** 2 + b ** 2)
    r = minimize(f, [1.0, 0.0], method="Nelder-Mead")
    return float(r.x[0]), (0.0 if slope_only else float(r.x[1]))


def wf_platt(df, pool, ev, lam, slope_only, renorm):
    """Target round r: fit on pool (earlier-season rows, honest) + this season's rounds < r."""
    out = df.copy()
    tot = {"win": 1.0, "podium": 3.0, "top10": 10.0}[ev]
    for r in sorted(df["round"].unique()):
        past = pd.concat([pool, df[df["round"] < r]]) if pool is not None else df[df["round"] < r]
        t = df["round"] == r
        if len(past) < 40:
            continue
        a, b = fit_platt(past[f"p_{ev}"].to_numpy(), past[f"y_{ev}"].to_numpy(), lam, slope_only)
        q = _sig(a * _logit(df.loc[t, f"p_{ev}"].to_numpy()) + b)
        if renorm:
            q = q / q.sum() * tot
        out.loc[t, f"p_{ev}"] = np.clip(q, 1e-4, 1 - 1e-4)
    return out


def wf_scale(variants, default, pool_variants, lam, grid):
    """Choose utility_scale per target round from `grid` by win log loss on earlier rounds (pool + this season),
    penalty lam*(log s)^2; return the stitched prediction frame (uses the cached walk-forward frames per scale)."""
    pieces = []
    for r in sorted(default["round"].unique()):
        best, bs = 1e18, 1.0
        for s in grid:
            v = variants[s]
            past = v[v["round"] < r]
            if pool_variants is not None:
                past = pd.concat([pool_variants[s], past])
            if past["round"].nunique() < 2:
                bs = 1.0; break
            q = np.clip(past["p_win"].to_numpy(), 1e-4, 1 - 1e-4); y = past["y_win"].to_numpy()
            val = -(y * np.log(q) + (1 - y) * np.log(1 - q)).sum() + lam * np.log(s) ** 2
            if val < best:
                best, bs = val, s
        pieces.append(variants[bs][variants[bs]["round"] == r].assign(chosen=bs))
    return pd.concat(pieces)


def report(name, season, df, base, rows):
    m = metrics(df); m.update(season=season, variant=name)
    for ev in EV:
        for kind in ("brier", "ll"):
            d, lo, hi = boot(df, base, ev, kind)
            m[f"d{kind}_{ev}"] = f"{d:+.4f}[{lo:+.4f},{hi:+.4f}]"
    rows.append(m)
    print(season, name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if k.startswith(("ll_", "winner", "win_hit", "pod_hit"))},
          m["dll_win"], m["dll_podium"], m["dll_top10"], flush=True)


def exp_B():
    cache = pd.read_pickle(OUT / "t063_cache.pkl")
    hist, _, _ = PR.load(True, 99)
    cfg0 = P.PredictorConfig(**{**PR.CONFIG.__dict__})
    # honest 2024 pool for the 2025 target (no 2025 data in its level-2 prior: w2025_level2=0)
    pool24 = run(hist, 2024, replace(cfg0, w2025_level2=0.0))
    pools = {2026: cache[(2025, "default")], 2025: pool24}
    rows = []
    for season in (2026, 2025):
        base = cache[(season, "default")]
        report("default", season, base, base, rows)
        for lam in (5, 30, 150):
            for slope_only in (True, False):
                for ev_set in ("win",):
                    # win only is recalibrated; podium/top10 left at default
                    for pooled in (False, True):
                        df = wf_platt(base, pools[season] if pooled else None, "win", lam, slope_only, renorm=True)
                        report(f"platt win lam={lam} {'slope' if slope_only else 'slope+int'} {'pooled' if pooled else 'own-season'}", season, df, base, rows)
        # in-model scale chosen walk-forward from cached scale frames
        grid = [0.85, 1.0, 1.15, 1.3, 1.5]
        var = {s: (base if s == 1.0 else cache[(season, f"scale={s}")]) for s in grid}
        pv = None
        for lam in (0, 20):
            df = wf_scale(var, base, pv, lam, grid)
            print("  chosen scales", season, lam, df.groupby("round")["chosen"].first().tolist())
            report(f"wf utility_scale (win LL, lam={lam}, own-season)", season, df, base, rows)
    pd.DataFrame(rows).to_csv(OUT / "t063_B.csv", index=False)


def exp_D():
    cache = pd.read_pickle(OUT / "t063_cache.pkl")
    hist, _, _ = PR.load(True, 99)
    res = hist["results"]
    for season in (2026, 2025):
        df = cache[(season, "default")]
        g = res[(res.season == season)][["round", "driver_id", "grid", "position", "team_id" if "team_id" in res else "driver_id"]].drop_duplicates(["round", "driver_id"])
        d = df.merge(g[["round", "driver_id", "grid"]], on=["round", "driver_id"], how="left")
        d["rank_in_race"] = d.groupby("round")["p_win"].rank(ascending=False)
        b = d[(d.p_win >= 0.10) & (d.p_win < 0.20)]
        print(f"\n== {season}: p_win in [0.10,0.20): n={len(b)}, mean p {b.p_win.mean():.3f}, wins {int(b.y_win.sum())}")
        print("  grid slot: mean", round(b.grid.mean(), 2), " median", b.grid.median(), " grid<=3:", int((b.grid <= 3).sum()), " grid>=8:", int((b.grid >= 8).sum()))
        print("  p_win rank in race:", b.rank_in_race.value_counts().sort_index().to_dict())
        print("  p_dnf mean", round(b.p_dnf.mean(), 3))
        by = b.groupby(pd.cut(b.grid, [0, 1, 3, 6, 10, 30])).agg(n=("y_win", "size"), p=("p_win", "mean"), obs=("y_win", "mean"))
        print(by.round(3).to_string())
        # all-driver p_win vs obs by grid bucket, rank-2+ drivers
        o = d[d.rank_in_race >= 2]
        print("  rank>=2 drivers, by grid bucket:")
        print(o.groupby(pd.cut(o.grid, [0, 1, 3, 6, 10, 30])).agg(n=("y_win", "size"), p=("p_win", "mean"), obs=("y_win", "mean"), pred_wins=("p_win", "sum"), wins=("y_win", "sum")).round(3).to_string())
        if season == 2026:
            print("  drivers in the 0.10-0.20 bin (code-ish rows):")
            print(b[["round", "driver_id", "grid", "p_win", "y_win"]].sort_values("round").to_string(index=False))


if __name__ == "__main__":
    if "B" in sys.argv[1:]: exp_B()
    elif "D" in sys.argv[1:]: exp_D()
    else: main()
