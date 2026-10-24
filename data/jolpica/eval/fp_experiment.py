"""T-041: does free-practice pace help the race predictor? Walk-forward, 2025 (tune) and 2026 (test).

Run from repo root: .venv/Scripts/python.exe data/jolpica/eval/fp_experiment.py [n_sims]
Needs data/jolpica/eval/fp_sessions_{2025,2026}.csv (fp_pull.py). Experiment only: the FP
table is injected through prepare_history(fp=...), predictor defaults are untouched.
Feature (predictor._fp_feature): -(team gap - min team gap) from a per-driver fp_gap; the level-2
PL fit learns b_fp from EARLIER races only (weekends without data contribute 0 = no information).
"""
import sys, itertools
from dataclasses import replace
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "backend"))
import predict_race as PR
from app.core import predictor as P

N_SIMS = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
CAP = 3.0  # seconds; clips the gap so a crashed/odd stint can't dominate


def sess_tables():
    s = pd.concat([pd.read_csv(HERE / f"fp_sessions_{y}.csv") for y in (2025, 2026)], ignore_index=True)
    s["wet"] = s["wet"].astype(bool)
    # sprint weekend = only one practice session happened (FP1)
    n = s.groupby("race_date")["session"].transform("nunique")
    s["sprint"] = n == 1
    return s


def to_fp(df, sched, col):
    d = df.dropna(subset=[col]).copy()
    d["fp_gap"] = d[col].clip(upper=CAP)
    sc = sched[["season", "round", "date"]]
    d = d.merge(sc, left_on="race_date", right_on="date", how="inner")
    return d[["season", "round", "code", "fp_gap"]]


def variants(s, sched):
    out = {}
    # V0 = T-027 feature: lap-weighted long-run gap over all sessions (incl. wet), reproduced
    v = s.dropna(subset=["lr_gap"]).groupby(["race_date", "code"]).apply(
        lambda x: pd.Series({"g": np.average(x["lr_gap"], weights=x["lr_n"])}), include_groups=False).reset_index()
    out["V0 all long-run"] = to_fp(v, sched, "g")
    dry = s[~s["wet"]]
    # C1 fastest lap: mean over dry sessions of (best lap - session's best lap); all weekends
    d = dry.dropna(subset=["best_lap"]).copy()
    d["g"] = d["best_lap"] - d.groupby("session_key")["best_lap"].transform("min")
    out["C1 fastest lap"] = to_fp(d.groupby(["race_date", "code"])["g"].mean().reset_index(), sched, "g")
    # C2 last dry session with long runs, normal (non-sprint) weekends only
    d = dry[~dry["sprint"]].dropna(subset=["lr_gap"]).copy()
    d["start"] = pd.to_datetime(d["start"], utc=True)
    last = d.groupby("race_date")["start"].transform("max")
    # last session that has >= 8 drivers with a long run
    cnt = d.groupby(["race_date", "session_key"])["code"].transform("count")
    d = d[cnt >= 8]
    d = d[d["start"] == d.groupby("race_date")["start"].transform("max")]
    out["C2 last dry session LR, no sprint"] = to_fp(d, sched, "lr_gap")
    return out


def run(hist, season, fp, mode, cfg, rounds):
    h = dict(hist)
    if fp is not None:
        f = fp.copy(); f["season"] = f["season"].astype(int); f["round"] = f["round"].astype(int)
        h["fp"] = f
    elif "fp" in h:
        del h["fp"]
    return P.walk_forward(h, season, rounds, f"model_{mode}", cfg)


def scale_by_stick(hist, fp, k, cfg):
    if k == 0:
        return fp
    f = fp.copy()
    m = {}
    for (se, r) in f[["season", "round"]].drop_duplicates().itertuples(index=False):
        m[(se, r)] = P.track_stickiness(hist, se, r, replace(cfg, stick_gamma=4.0))["mult"]
    f["fp_gap"] = f["fp_gap"] * np.array([m[(a, b)] ** k for a, b in zip(f["season"], f["round"])])
    return f


def main():
    hist, _, sched = PR.load(True, PR.next_round(True))
    sched = sched[["season", "round", "date"]]
    s = sess_tables()
    var = variants(s, sched)
    res = hist["results"]
    rows, preds = [], {}
    base = dict(n_sims=N_SIMS)
    cfgs = {2025: P.PredictorConfig(**base, w2025_level2=0.0, driver_prior_2025=0.0),
            2026: P.PredictorConfig(**base)}
    # fp coverage
    for se in (2025, 2026):
        last = int(res.query("season == @se")["round"].max())
        cov = {n: v[v["season"] == se]["round"].nunique() for n, v in var.items()}
        print(se, "races", last - 1, "(R2-%d)" % last, "FP weekends with data:", cov)

    def record(se, mode, name, fp, k=0):
        last = int(res.query("season == @se")["round"].max())
        cfg = replace(cfgs[se], use_fp=fp is not None)
        f = None if fp is None else scale_by_stick(hist, fp, k, cfgs[se])
        wf = run(hist, se, f, mode, cfg, range(2, last + 1))
        sc = P.score(wf)
        preds[(se, mode, name)] = wf
        rows.append({"season": se, "mode": mode, "variant": name, **sc})
        print(se, mode, name, {k_: round(v_, 4) for k_, v_ in sc.items() if k_ in ("win_logloss", "podium_logloss", "top10_logloss", "winner_nll")}, flush=True)

    for se in (2025, 2026):
        for mode in ("prequali", "grid"):
            record(se, mode, "current (no FP)", None)
            for n, fp in var.items():
                record(se, mode, n, fp)
    # C3: V0 x stickiness^k, k tuned on 2025 (both modes, pick by 2025 win+podium LL sum)
    ks = (-2, -1, 1, 2)
    for se in (2025, 2026):
        for mode in ("prequali", "grid"):
            for k in ks:
                record(se, mode, f"C3 V0 x stick^{k}", var["V0 all long-run"], k)
    t = pd.DataFrame(rows)
    t.to_csv(HERE / "fp_experiment.csv", index=False)
    pd.concat([d.assign(season=k[0], mode=k[1], variant=k[2]) for k, d in preds.items()]).to_csv(HERE / "fp_experiment_predictions.csv", index=False)
    pd.set_option("display.width", 220)
    print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
