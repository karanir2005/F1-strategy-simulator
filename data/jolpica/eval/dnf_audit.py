"""T-045: DNF model audit. Standalone re-implementation of predictor.dnf_probabilities with extra
knobs (recency half-life, driver/team shrinkage), scored on every started entry of a season's
rounds 2..N (p_dnf from strictly earlier rounds of the same season). Tune on 2025, test on 2026.

    python data/jolpica/eval/dnf_audit.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "backend"))
import numpy as np
import pandas as pd
from app.core import predictor as P

R = pd.read_csv(ROOT / "data" / "jolpica" / "derived" / "results.csv")
R = R[R["session"] == "race"] if "session" in R.columns else R
R["classified"] = R["classified"].astype(bool)
R = R[P._started(R)].copy()  # drop DNS: never raced
R["nf"] = (~R["classified"]).astype(float)


def p_dnf(season, rnd, ent, prior_rate=0.15, prior_n=22.0, team_m=10.0, driver_m=20.0, halflife=None, team_only=False):
    past = R[(R["season"] == season) & (R["round"] < rnd)]
    w = np.ones(len(past)) if halflife is None else 0.5 ** ((rnd - 1 - past["round"].to_numpy()) / halflife)
    past = past.assign(w=w, wnf=w * past["nf"].to_numpy())
    field = (past["wnf"].sum() + prior_rate * prior_n) / (past["w"].sum() + prior_n)
    tg, dg = past.groupby("team_id"), past.groupby("driver_id")
    out = []
    for d, t in zip(ent["driver_id"], ent["team_id"]):
        tn, tk = tg["w"].sum().get(t, 0.0), tg["wnf"].sum().get(t, 0.0)
        tr = (tk + team_m * field) / (tn + team_m)
        if team_only:
            out.append(tr)
            continue
        dn, dk = dg["w"].sum().get(d, 0.0), dg["wnf"].sum().get(d, 0.0)
        out.append((dk + driver_m * tr) / (dn + driver_m))
    return np.array(out)


def table(season, **kw):
    rows = []
    for rnd in sorted(R[R["season"] == season]["round"].unique()):
        if rnd < 2:
            continue
        ent = R[(R["season"] == season) & (R["round"] == rnd)]
        rows.append(ent.assign(p=p_dnf(season, rnd, ent, **kw)))
    return pd.concat(rows)


def score(t):
    p, y = np.clip(t["p"].to_numpy(), 1e-4, 1 - 1e-4), t["nf"].to_numpy()
    return dict(ll=float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()), brier=float(((p - y) ** 2).mean()),
                mean_p=float(p.mean()), obs=float(y.mean()), n=len(t))


VARIANTS = {
    "current": dict(),
    "constant field rate": dict(team_m=1e9, driver_m=1e9),
    "team-only (m=10)": dict(team_only=True),
    "A stronger shrink (team 20, drv 60)": dict(team_m=20.0, driver_m=60.0),
    "B recency hl=6": dict(halflife=6.0),
    "C team-only + hl=6": dict(team_only=True, halflife=6.0),
    "D shrink(20,60)+hl=6": dict(team_m=20.0, driver_m=60.0, halflife=6.0),
    "E drv 40": dict(driver_m=40.0),
}

if __name__ == "__main__":
    rows = []
    for name, kw in VARIANTS.items():
        for s in (2025, 2026):
            rows.append(dict(variant=name, season=s, **score(table(s, **kw))))
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(out.round(4).to_string(index=False))
    # calibration of the current model
    for s in (2025, 2026):
        t = table(s)
        b = pd.cut(t["p"], [0, 0.08, 0.12, 0.16, 0.2, 0.3, 1.0])
        print(f"\ncurrent, {s}")
        print(t.groupby(b, observed=True).agg(n=("p", "size"), pred=("p", "mean"), obs=("nf", "mean")).round(3).to_string())
    t = table(2026)
    print("\n2026 by team (mean pred vs observed, rounds 2-18):")
    print(t.groupby("team").agg(n=("p", "size"), pred=("p", "mean"), obs=("nf", "mean")).round(3).sort_values("obs").to_string())
    print("\n2026 by round (field obs rate):", R[R.season == 2026].groupby("round")["nf"].mean().round(2).tolist())
    print("2025 by round:", R[R.season == 2025].groupby("round")["nf"].mean().round(2).tolist())
