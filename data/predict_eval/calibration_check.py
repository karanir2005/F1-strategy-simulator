"""
T-057 (2026-11-16): probability calibration check for the race predictor (grid mode, as shipped).

Run from the repo root:   .venv\\Scripts\\python data\\predict_eval\\calibration_check.py [--seasons 2026 2025]

Everything is in memory: it calls predictor.walk_forward through the same loaders as predict_race.py
(offline, cached Jolpica data) and writes only to data/predict_eval/out/. It does NOT call
`predict_race.py --by-race`, which ignores --out-dir and overwrites the shipped artifacts.
Predictions are strictly walk-forward (each race from earlier rounds + its own grid).

What it reports, for p_win / p_podium / p_top10, pooled over driver-races:
  * reliability table (bins, mean predicted, observed rate, count, Wilson 95% interval)
  * Brier, log loss, expected calibration error (count-weighted |pred - obs| over the bins) for the
    model vs. the grid-order baseline (predictor.RankBaseline 'grid', which is handed 4 extra seasons)
  * paired race-bootstrap interval for the Brier difference (model - baseline)
  * walk-forward recalibration: logit temperature  p' = sigmoid(logit(p)/T)  and, for p_win only, a
    power-renormalised version  p' = p^a / sum(p^a). T / a are fitted by minimising log loss on rounds
    BEFORE the target round only (identity until 2 earlier races exist), then applied to the target.
  * T fitted on all of 2025 applied to 2026 (cross-season check), when both seasons are run.

Caveat baked into every number: the 22 drivers of one race are not independent (exactly one wins), so the
Wilson intervals are optimistic; 20 races is ~440 driver-races and only 20 wins.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

import predict_race as PR
from app.core import predictor as P

OUT = Path(__file__).resolve().parent / "out"
EVENTS = ("win", "podium", "top10")
BINS = {
    "win": [0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 1.0001],
    "podium": [0, 0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95, 1.0001],
    "top10": [0, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95, 0.99, 1.0001],
}
EPS = 1e-4


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def logloss(p, y):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


def reliability(p, y, ev):
    b = pd.cut(p, BINS[ev], right=False)
    df = pd.DataFrame({"p": p, "y": y, "bin": b})
    rows = []
    for k, g in df.groupby("bin", observed=True):
        lo, hi = wilson(g["y"].sum(), len(g))
        rows.append({"bin": f"[{k.left:.2f},{min(k.right, 1):.2f})", "n": len(g), "mean_pred": g["p"].mean(),
                     "obs_rate": g["y"].mean(), "wilson_lo": lo, "wilson_hi": hi})
    return pd.DataFrame(rows)


def ece(p, y, ev):
    t = reliability(p, y, ev)
    return float((t["n"] * (t["mean_pred"] - t["obs_rate"]).abs()).sum() / t["n"].sum())


def temp_apply(p, T):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    z = np.log(p / (1 - p)) / T
    return 1 / (1 + np.exp(-z))


def fit_T(p, y):
    f = lambda lt: logloss(temp_apply(p, np.exp(lt)), y)
    return float(np.exp(minimize_scalar(f, bounds=(np.log(0.4), np.log(4.0)), method="bounded").x))


def power_apply(df, a, ev):
    """p^a renormalised within each race to the event total (1 / 3 / 10)."""
    tot = {"win": 1.0, "podium": 3.0, "top10": 10.0}[ev]
    q = np.clip(df[f"p_{ev}"], EPS, 1) ** a
    return q / q.groupby(df["round"]).transform("sum") * tot


def fit_power(df, ev):
    y = df[f"y_{ev}"].to_numpy()
    f = lambda a: logloss(power_apply(df, a, ev).to_numpy(), y)
    return float(minimize_scalar(f, bounds=(0.4, 3.0), method="bounded").x)


def walk_forward_recal(df, ev, min_prior=2):
    """Per target round: fit on strictly earlier rounds of the same season."""
    tp = pd.Series(np.nan, index=df.index)
    pp = pd.Series(np.nan, index=df.index)
    Ts = {}
    for r in sorted(df["round"].unique()):
        past = df[df["round"] < r]
        tgt = df["round"] == r
        if past["round"].nunique() >= min_prior:
            T = fit_T(past[f"p_{ev}"].to_numpy(), past[f"y_{ev}"].to_numpy())
            a = fit_power(past, ev) if ev == "win" else 1.0
        else:
            T, a = 1.0, 1.0
        Ts[r] = (T, a)
        tp[tgt] = temp_apply(df.loc[tgt, f"p_{ev}"].to_numpy(), T)
        pp[tgt] = power_apply(df[tgt], a, ev) if ev == "win" else df.loc[tgt, f"p_{ev}"]
    return tp, pp, Ts


def boot_diff(a, b, y, rounds, n=4000, seed=0):
    """Paired race-level bootstrap 95% interval of mean sq err(a) - mean sq err(b)."""
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"r": rounds, "ba": (np.asarray(a) - y) ** 2, "bb": (np.asarray(b) - y) ** 2})
    g = d.groupby("r")[["ba", "bb"]].sum().to_numpy()
    cnt = d.groupby("r").size().to_numpy()
    idx = np.arange(len(g))
    out = []
    for _ in range(n):
        s = rng.choice(idx, len(idx))
        out.append((g[s, 0].sum() - g[s, 1].sum()) / cnt[s].sum())
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def get_predictions(season):
    hist, _, _ = PR.load(True, 99)
    res = hist["results"]
    last = int(res[res["season"] == season]["round"].max())
    cfg = P.PredictorConfig(**{**PR.CONFIG.__dict__})
    model = P.walk_forward(hist, season, range(1, last + 1), "model_grid", cfg)
    base = P.walk_forward(hist, season, range(2, last + 1), "grid", cfg)
    return model, base, last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+", default=[2026, 2025])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    summary, preds = [], {}
    for season in args.seasons:
        print(f"\n================ {season} ================")
        model, base, last = get_predictions(season)
        preds[season] = model
        model.to_csv(OUT / f"walk_forward_model_grid_{season}.csv", index=False)
        m2 = model[model["round"] >= 2].reset_index(drop=True)
        b2 = base.set_index(["round", "driver_id"]).loc[list(zip(m2["round"], m2["driver_id"]))].reset_index()
        print(f"{model['round'].nunique()} races R1-{last} (R2-{last} for model-vs-baseline); {len(model)} driver-races")
        for ev in EVENTS:
            y1, p1 = model[f"y_{ev}"].to_numpy(), model[f"p_{ev}"].to_numpy()
            y, p, pb = m2[f"y_{ev}"].to_numpy(), m2[f"p_{ev}"].to_numpy(), b2[f"p_{ev}"].to_numpy()
            print(f"\n--- {season} {ev}: reliability (model, R1-{last}) ---")
            rel = reliability(p1, y1, ev)
            print(rel.round(3).to_string(index=False))
            rel.assign(season=season, event=ev).to_csv(OUT / f"reliability_{season}_{ev}.csv", index=False)
            lo, hi = boot_diff(p, pb, y, m2["round"].to_numpy())
            tp, pp, Ts = walk_forward_recal(m2, ev)
            tpa = tp.to_numpy()
            row = {"season": season, "event": ev, "n_races": m2["round"].nunique(), "n": len(m2),
                   "mean_p": p.mean(), "obs_rate": y.mean(),
                   "brier_model": brier(p, y), "brier_grid_base": brier(pb, y),
                   "brier_diff_ci95": f"[{lo:+.4f},{hi:+.4f}]",
                   "ll_model": logloss(p, y), "ll_grid_base": logloss(pb, y),
                   "ece_model": ece(p, y, ev), "ece_grid_base": ece(pb, y, ev),
                   "temp_brier": brier(tpa, y), "temp_ll": logloss(tpa, y), "temp_ece": ece(tpa, y, ev),
                   "T_full_sample": fit_T(p, y)}
            held = (m2["round"] >= 6).to_numpy()
            row["brier_model_R6+"] = brier(p[held], y[held])
            row["temp_brier_R6+"] = brier(tpa[held], y[held])
            row["ll_model_R6+"] = logloss(p[held], y[held])
            row["temp_ll_R6+"] = logloss(tpa[held], y[held])
            if ev == "win":
                ppa = pp.to_numpy()
                row.update({"power_brier": brier(ppa, y), "power_ll": logloss(ppa, y), "a_full_sample": fit_power(m2, ev)})
            row["T_last5_rounds"] = " ".join(f"{Ts[r][0]:.2f}" for r in sorted(Ts)[-5:])
            summary.append(row)
        fav = model.sort_values("p_win", ascending=False).groupby("round").head(1)
        print(f"\nfavourite: mean P(win) {fav['p_win'].mean():.3f} vs observed win rate {fav['y_win'].mean():.3f} (n={len(fav)})")
        summary.append({"season": season, "event": "favourite", "n": len(fav), "mean_p": fav["p_win"].mean(), "obs_rate": fav["y_win"].mean()})

    if 2025 in preds and 2026 in preds:
        print("\n===== cross-season: T fitted on all of 2025, applied to 2026 R2+ =====")
        a, b = preds[2025], preds[2026]
        b = b[b["round"] >= 2]
        for ev in EVENTS:
            T = fit_T(a[f"p_{ev}"].to_numpy(), a[f"y_{ev}"].to_numpy())
            y, p = b[f"y_{ev}"].to_numpy(), b[f"p_{ev}"].to_numpy()
            q = temp_apply(p, T)
            r = {"season": "2025->2026", "event": ev, "T_fit_2025": T, "brier_model": brier(p, y), "temp_brier": brier(q, y),
                 "ll_model": logloss(p, y), "temp_ll": logloss(q, y)}
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
            summary.append(r)
    s = pd.DataFrame(summary)
    s.to_csv(OUT / "calibration_summary.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 50)
    print("\n=== SUMMARY ===")
    print(s.round(4).T.to_string())


if __name__ == "__main__":
    main()
