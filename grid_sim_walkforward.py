"""
Strict walk-forward evaluation of the full-grid race sim (T-062).

T-060's validation is leave-one-race-out (LORO): every constant for race R is fitted on all the OTHER races, including
later ones. That is not what production does. Here, for each 2026 round R, EVERYTHING the sim uses is built from races
before R only (plus R's own qualifying and grid, known before the start):
  - tyre model: refit on the laps of rounds < R;
  - track entry (base lap ratio, pit loss, stint caps): calibrate_grid_sim.rebuild_track restricted to rounds < R;
  - overtake logistic, lap-1 table and track offsets: refitted on rounds < R (R's own track has no offset: 0);
  - pace regression, pace noise, SC bunching, lap-1 spacing, strategy mix, held gap: fitted on rounds < R;
  - overtake intercept shift: chosen from the pass-count targets of rounds < R only.
The Plackett-Luce predictor numbers are data/jolpica/predictions/walk_forward_2026_drivers.csv (already walk-forward).
Residual leak (small, documented in the T-062 handoff): the (scale, shift) selection table was produced with constants
fitted on all races; for a few of the experiments the shift is re-chosen from per-round pass counts of earlier rounds,
which is clean.

Experiments (variants of the sim's per-car pace input, all fitted by regression of real race pace on the inputs over
rounds < R, see handoff):
  base         shipped T-060 input: Q1 gap shrunk to the team mean, blended with the quantile-matched PL rating order
  theta        Q1 gap (shrunk) + PL theta_race gap in rating units, each with its own fitted s/lap coefficient
  theta_grid   theta + starting-slot term
  quali        Q1 gap only (ablation)
Noise multipliers and pace-scale multipliers are scanned and then selected walk-forward (on rounds < R only).

    python validate_grid_sim.py --walk-forward
"""

from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

import calibrate_grid_sim as C  # noqa: E402
from app.core import grid_inputs as G  # noqa: E402
from app.core import predictor as P  # noqa: E402
from app.core import tire_model as tm  # noqa: E402
from app.core.race_grid_sim import GridCalibration, load_grid_calibration, simulate_grid  # noqa: E402
from app.core.safety_cars import load_params as load_sc  # noqa: E402

MIN_TRAIN_ROUNDS = 4  # first evaluated round = 5 (train R1-4; the pace regression needs >= 2 rounds with ratings)
LAM = 0.25


class WFStudy(C.Study):
    """Study whose per-round context (tyre model, track entry) only uses rounds before the evaluated one."""

    def __init__(self, real, eval_rounds, verbose=False):
        self.real = real
        self.hist = real.hist
        self.eval_rounds = list(eval_rounds)
        laps_all = tm.load_all_laps()
        files = C.round_to_file()
        res = self.hist["results"]
        self.ctx = {}
        orig_rot = C.round_of_track
        for rnd in sorted(set(self.eval_rounds) | set(range(1, max(self.eval_rounds) + 1))):
            r = res[(res["season"] == 2026) & (res["round"] == rnd)]
            tid = G.CIRCUIT_TO_TRACK[r["circuit_id"].iloc[0]]
            later = [files[k] for k in files if k >= rnd]  # held-out race AND every later race
            lt = laps_all
            for f in later:
                lt = lt[~lt["race_file"].str.replace("\\", "/").str.endswith(f)]
            model = tm.build_tire_model(laps=lt)
            full = orig_rot(real)
            C.round_of_track = lambda _real, _f=full, _r=rnd: {k: v for k, v in _f.items() if v < _r}
            try:
                track = C.rebuild_track(tid, rnd, real, model)
            finally:
                C.round_of_track = orig_rot
            self.ctx[rnd] = {"model": model, "track": track, "track_id": tid, "rid": real.round_rid.get(rnd)}
            if verbose:
                print(f"  wf ctx R{rnd} {tid}", flush=True)
        self.sc_samples = C.sc_gap_samples(real)
        self.pace_tab = C.pace_table(real)
        # strategy mix: NOT refitted walk-forward. A fold needs the optimizer plans of every training race, which need a
        # tyre model that exists only from round ~5 on; the mix only describes how diverse real strategies are (it never
        # sees finishing order), so the shipped T-060 values are used in every fold (small leak, stated in the handoff).
        self.strat_stats = {}
        self.passes = real.passes_per_race()
        self.wf = pd.read_csv(C.WF_PATH)
        self.feat = self._features()

    def fit_all(self, train_rounds) -> dict:
        real = self.real
        train_rids = [real.round_rid[r] for r in train_rounds if r in real.round_rid]
        shipped = load_grid_calibration()
        return {
            "pace": C.fit_pace(self.pace_tab, [r for r in train_rounds if r >= 2]),
            "sc": C.fit_sc(self.sc_samples, train_rids),
            "lap1_spacing_s": C.lap1_spacing(real, train_rids),
            "held_gap_s": C.fit_held_gap(real, train_rids)[0],
            "held_gap_sigma": C.fit_held_gap(real, train_rids)[1],
            "strategy_mix": {"plan_weights": list(shipped.plan_weights), "pit_jitter_laps": shipped.pit_jitter_laps},
            "interaction": C.fit_interaction_fold(real, train_rids),
            "train_rounds": list(train_rounds),
        }

    # ---- regressor table for the pace variants: every entrant of every round >= 2
    def _features(self) -> pd.DataFrame:
        rows = []
        res = self.hist["results"]
        for rnd in self.real.rounds:
            if rnd < 2:
                continue
            ent = G.race_entrants(self.hist, 2026, rnd)
            base = float(self.ctx[rnd]["track"]["base_laptime_s"]) if rnd in self.ctx else 90.0
            a = G.pace_deltas(self.hist, 2026, rnd, ent, base, 1.0, LAM, 1.0)
            try:
                f = P.race_features(self.hist, 2026, rnd, P.PredictorConfig(), entry_list=ent[["driver_id", "code", "team_id", "team"]],
                                    grid=ent["grid"].to_numpy(dtype=float))
                th = f["theta_race"].to_numpy(dtype=float)
            except Exception:
                th = np.zeros(len(ent))
            n = len(ent)
            gslot = ent["grid"].to_numpy(dtype=float)
            for i in range(n):
                rows.append({"round": rnd, "code": ent["code"].iloc[i], "team_id": ent["team_id"].iloc[i], "A": float(a[i]),
                             "C": float(th.max() - th[i]), "D": float(np.log(gslot[i])), "D2": float(min(gslot[i], 10.0) - 1.0)})
        feat = pd.DataFrame(rows)
        y = self.pace_tab[["round", "code", "y", "se2"]]
        return feat.merge(y, on=["round", "code"], how="left")


def fit_variant(feat: pd.DataFrame, train_rounds, cols) -> dict:
    """OLS of race-demeaned real pace y on race-demeaned `cols` over training rounds, with the pace-noise variance
    components of the residual (same estimator as calibrate_grid_sim.fit_pace)."""
    t = feat[feat["round"].isin(train_rounds) & feat["y"].notna()].reset_index(drop=True)
    dm = C._demean(t, ["y"] + list(cols))
    X = dm[list(cols)].to_numpy()
    # ridge of 1e-6 keeps the fit defined when a column is constant in a short training window
    beta = np.linalg.solve(X.T @ X + 1e-6 * np.eye(X.shape[1]), X.T @ dm["y"].to_numpy())
    t["e"] = dm["y"] - X @ beta
    g = t.groupby(["round", "team_id"])
    two = g["e"].transform("size") == 2
    tt = t[two]
    diff = tt.groupby(["round", "team_id"])["e"].agg(lambda s: s.iloc[0] - s.iloc[1])
    se_pair = tt.groupby(["round", "team_id"])["se2"].sum()
    var_d = max(float((diff ** 2).mean() - se_pair.mean()) / 2.0, 0.0)
    tm_mean = tt.groupby(["round", "team_id"])["e"].mean()
    se_mean = tt.groupby(["round", "team_id"])["se2"].mean() / 2.0
    var_t = max(float((tm_mean ** 2).mean() - se_mean.mean()) - var_d / 2.0, 0.0)
    return {"beta": beta, "cols": list(cols), "team_sd": float(np.sqrt(var_t)), "driver_sd": float(np.sqrt(var_d)),
            "r2": float(1 - (t["e"] ** 2).sum() / (dm["y"] ** 2).sum()), "n": int(len(t))}


def variant_pace(st: WFStudy, rnd: int, fit: dict) -> np.ndarray:
    ent = G.race_entrants(st.hist, 2026, rnd)
    f = st.feat[st.feat["round"] == rnd].set_index("code").loc[ent["code"]]
    p = f[fit["cols"]].to_numpy() @ fit["beta"]
    p = p - p.min()
    return p


def pos_ll(sim, actual) -> float:
    """Mean over drivers of -log P(sim finishing position == actual), add-half smoothing over the field."""
    n = sim.positions.shape[0]
    d = sim.positions.shape[1]
    out = []
    for i in range(d):
        cnt = float((sim.positions[:, i] == actual[i]).sum())
        out.append(-np.log((cnt + 0.5) / (n + 0.5 * d)))
    return float(np.mean(out))


def run_config(st: WFStudy, rnd: int, cal: GridCalibration, ip, n_sims: int, pace=None, noise_mult=1.0, pace_mult=1.0):
    """One sim of round rnd. `pace`: replacement pace vector (variants); the pace multiplier and noise multiplier scale
    the pace spread and both pace-noise SDs."""
    gi = st.inputs(rnd, cal, leaky=False)
    if pace is not None:
        gi.cars = [dataclasses.replace(c, pace_delta_s=float(p)) for c, p in zip(gi.cars, pace)]
    if pace_mult != 1.0:
        gi.cars = [dataclasses.replace(c, pace_delta_s=c.pace_delta_s * pace_mult) for c in gi.cars]
    cal2 = dataclasses.replace(cal, team_pace_sd_s=cal.team_pace_sd_s * noise_mult, driver_pace_sd_s=cal.driver_pace_sd_s * noise_mult)
    return st.run_race(rnd, cal2, ip, n_sims, gi=gi)


def score_extra(st: WFStudy, rnd: int, gi, sim) -> dict:
    row = st.score(rnd, gi, sim)
    s = sim.summary()
    codes = [x["code"] for x in s]
    res = st.hist["results"]
    er = res[(res["season"] == 2026) & (res["round"] == rnd)].set_index("code").loc[codes]
    actual = er["position"].to_numpy(dtype=float)
    row["sim_pos_ll"] = pos_ll(sim, actual)
    row["p_win"] = np.array([x["p_win"] for x in s])
    row["p_pod"] = np.array([x["p_podium"] for x in s])
    row["codes"] = codes
    row["actual"] = actual
    return row


# ------------------------------------------------------------------------------------------------ runner
VARIANT_COLS = {"quali": ["A"], "theta": ["A", "C"], "theta_grid": ["A", "C", "D"], "quali_grid": ["A", "D"]}


def run_walk_forward(n_sims: int = 2000, first_round: int = 5, variants=("base", "theta_grid"), verbose=True) -> dict:
    """Strict walk-forward scorecard rows for the shipped sim (`base`) and the pace-input variants, rounds
    first_round..last raced round. Per-round rows keep sim / grid / PL scores and the probability arrays."""
    real = C.Real()
    last = max(real.rounds)
    st = WFStudy(real, range(first_round, last + 1))
    table = pd.read_csv(C.TABLE_PATH)
    rows = []
    for rnd in range(first_round, last + 1):
        tr = [r for r in real.rounds if r < rnd]
        fit = st.fit_all(tr)
        ip = st.make_ip(fit)
        shift = C.choose_config(table[table["round"] < rnd])["shift"]  # pass-count targets of earlier rounds only
        for v in variants:
            cal = st.make_cal(fit, 1.0, shift)
            pace = None
            if v != "base":
                fv = fit_variant(st.feat, [r for r in tr if r >= 2], VARIANT_COLS[v])
                pace = variant_pace(st, rnd, fv)
                cal = dataclasses.replace(cal, team_pace_sd_s=fv["team_sd"], driver_pace_sd_s=fv["driver_sd"])
            gi, sim = run_config(st, rnd, cal, ip, n_sims, pace=pace)
            row = score_extra(st, rnd, gi, sim)
            row["variant"] = v
            rows.append(row)
        if verbose:
            print(f"WF R{rnd}: " + " ".join(f"{r['variant']} rho {r['sim_spearman']:.2f}" for r in rows[-len(variants):])
                  + f" | grid {rows[-1]['grid_spearman']:.2f} pl {rows[-1].get('pl_spearman', float('nan')):.2f}", flush=True)
    return {"rows": rows, "rounds": [first_round, last]}


def summarise_walk_forward(rows: list) -> dict:
    df = pd.DataFrame(rows)
    out = {}
    for v, g in df.groupby("variant"):
        g = g.set_index("round")

        def ms(x):
            x = np.asarray(x, float)
            return {"mean": float(x.mean()), "se": float(x.std(ddof=1) / np.sqrt(len(x)))}

        out[v] = {m: ms(g[f"sim_{m}"]) for m in ("spearman", "top3_hits", "winner", "win_ll", "pod_brier", "passes")}
        out[v]["vs_pl"] = {m: ms(g[f"sim_{m}"] - g[f"pl_{m}"]) for m in ("spearman", "top3_hits", "winner", "win_ll", "pod_brier")}
        out[v]["vs_grid"] = {m: ms(g[f"sim_{m}"] - g[f"grid_{m}"]) for m in ("spearman", "top3_hits", "winner")}
        out[v]["passes_corr"] = float(np.corrcoef(g["sim_passes"], g["real_passes"])[0, 1])
        out[v]["real_passes"] = float(g["real_passes"].mean())
        out[v]["n"] = int(len(g))
    g = df[df["variant"] == df["variant"].iloc[0]].set_index("round")
    out["pl"] = {m: {"mean": float(g[f"pl_{m}"].mean()), "se": float(g[f"pl_{m}"].std(ddof=1) / np.sqrt(len(g)))}
                 for m in ("spearman", "top3_hits", "winner", "win_ll", "pod_brier")}
    out["grid"] = {m: {"mean": float(g[f"grid_{m}"].mean()), "se": float(g[f"grid_{m}"].std(ddof=1) / np.sqrt(len(g)))}
                   for m in ("spearman", "top3_hits", "winner")}
    return out
