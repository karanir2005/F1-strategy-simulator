"""
Calibration of the full-grid race simulation (T-060): fit the constants that T-034 left as PLACEHOLDERs from real 2026
data, leave-one-race-out (LORO) only, and write data/sim_params/grid_sim_calibration.json.

Everything here is a function of a set of TRAINING races, so the same code produces (a) the shipped constants (all
races) and (b) the fold constants used by validate_grid_sim.py (all races but the held-out one). Nothing is tuned on
the finishing-order scorecard: each constant is fitted against its own real-data target.

What is fitted and against what (long version, 2026-11-16; see team/handoffs/T-060-data-scientist.md for numbers):

1. Race pace vs qualifying (grid_inputs.RACE_PACE_SCALE, DRIVER_SHRINK, QUALI_WEIGHT). Target: each driver's real
   tyre/fuel-normalised race pace (clean-air laps, driver fixed effect with race-lap fixed effects, as in
   fit_race_interaction.driver_race_pace but also returning its sampling variance). Regressors: the sim's own pace
   inputs at scale 1 (Q1 gap shrunk to the team mean, and the walk-forward rating order quantile-matched onto the Q1
   gaps), per race, race-demeaned. lambda (DRIVER_SHRINK) by grid search on SSE, then OLS y ~ A + B gives the scale
   (A+B) and the quali weight (A/(A+B)). No finishing-order information is used.
2. Race-day pace uncertainty (TEAM_PACE_SD_S, DRIVER_PACE_SD_S): variance components of the regression residual
   (team mean / within-team difference), minus the sampling variance of the pace estimates themselves.
3. SC bunching (SC_GAP_S, SC_GAP_CLOSURE): car-to-car gaps behind the car ahead (same lap, not pitting) by SC lap in
   real SC episodes: the floor is the median gap late in long episodes, the closure is the per-lap geometric decay of
   the median gap from lap 2 of an episode on.
4. Lap-1 spacing (LAP1_SPACING_S): slope of gap-to-leader at the end of lap 1 against position - 1, median over races.
5. Pit strategy mix (PLAN_WEIGHTS, PIT_JITTER_LAPS): each real driver strategy is assigned to the nearest of the
   optimizer's top-3 plans (distance = stop-count mismatch + stop-lap differences + compound mismatches), weights =
   shares; jitter = per-stop spread of the matched stop laps, converted to the half-width of the uniform jitter.
   This only describes the field's plan diversity, the optimizer's plans are not tuned to it.
6. Overtake calibration (pace_delta_scale, intercept_shift). T-053 fitted the logistic on in-race (hindsight) pace
   and per-race passes were 13/race; the sim uses qualifying-based pace and made ~60. The scale multiplies the pace
   delta inside the logit and the shift moves the intercept; the pair is chosen so the sim's passes per race and its
   spread of grid->finish position change match the real ones (stats of passes and of position change only,
   never finishing-order accuracy), by a grid search whose targets exclude the held-out race.
7. (validation) overtake logistic + lap-1 table are refitted per fold with the held-out race removed (cheap), the
   held-out track offset is dropped, and track_params (base lap, pit loss, stints) are rebuilt without the held-out
   race. See validate_grid_sim.py.

    python calibrate_grid_sim.py            # fit on all races, write data/sim_params/grid_sim_calibration.json
    python calibrate_grid_sim.py --no-write
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

import fit_race_interaction as F  # noqa: E402
from app.core import grid_inputs as G  # noqa: E402
from app.core import tire_model as tm  # noqa: E402
from app.core.track_params import get_track  # noqa: E402

OUT_PATH = ROOT / "data" / "sim_params" / "grid_sim_calibration.json"
TABLE_PATH = ROOT / "data" / "scorecard" / "grid_sim_selection_table.csv"
LAMBDA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)
MIN_CLEAN_LAPS = 8


# --------------------------------------------------------------------------------------------- real data
class Real:
    """Everything measured from the real 2026 races, built once."""

    def __init__(self):
        self.hist = G.load_history()
        laps = F.add_gap_columns(F.load_races(2026))
        self.laps = laps
        self.rid_round = {}
        for d in (ROOT / "data" / "processed", ROOT / "data" / "staging"):
            for p in d.glob("*_2026_race_info.json"):
                rid = p.name.replace("_race_info.json", "")
                info = json.loads(p.read_text(encoding="utf-8"))
                if info.get("round") and rid not in self.rid_round:
                    self.rid_round[rid] = int(info["round"])
        self.races = sorted((r for r in laps["race_id"].unique() if r in self.rid_round), key=lambda r: self.rid_round[r])
        self.round_rid = {self.rid_round[r]: r for r in self.races}
        self.rounds = sorted(self.round_rid)
        self.pace = driver_pace_with_se(laps)
        self.events = self._events()

    def _events(self):
        ev = F.build_pass_events(self.laps, F.OVERTAKE_WINDOW_S)
        tid = self.laps.drop_duplicates("race_id").set_index("race_id")["track_id"]
        ev["track_id"] = ev["race_id"].map(tid)
        pace = self.pace.set_index(["race_id", "driver"])["y"]
        a = [pace.get((r, d), np.nan) for r, d in zip(ev["race_id"], ev["driver"])]
        b = [pace.get((r, d), np.nan) for r, d in zip(ev["race_id"], ev["defender"])]
        ev["pace_delta"] = np.clip(np.array(b) - np.array(a), -F.PACE_DELTA_CLIP_S, F.PACE_DELTA_CLIP_S)
        return ev

    def passes_per_race(self) -> pd.Series:
        """Real on-track passes per race, counted as T-053 counts them (all events, pace known or not)."""
        s = self.events.groupby("race_id")["passed"].sum()
        return pd.Series({self.rid_round[r]: float(s.get(r, 0.0)) for r in self.races})

    def position_change(self, rnd: int, ent: pd.DataFrame):
        """(sd, mean abs) of (grid slot - official finishing position) over CLASSIFIED drivers of a race."""
        res = self.hist["results"]
        r = res[(res["season"] == 2026) & (res["round"] == rnd)].set_index("code")
        e = ent.set_index("code")
        d = pd.DataFrame({"grid": e["grid"], "pos": r["position"], "cls": r["classified"]}).dropna()
        d = d[d["cls"].astype(bool)]
        ch = (d["grid"] - d["pos"]).to_numpy(float)
        return float(ch.std(ddof=1)), float(np.abs(ch).mean())


def driver_pace_with_se(laps: pd.DataFrame) -> pd.DataFrame:
    """fit_race_interaction.driver_race_pace plus the sampling variance of each estimate (resid var / n laps)."""
    clean = tm.clean_laps(laps)
    clean = clean[clean["ahead_gap"].fillna(0) >= 3.0]
    clean = clean[clean["ahead_gap"].notna()].copy()
    med = clean.groupby("race_id")["lap_time_seconds"].transform("median")
    clean = clean[clean["lap_time_seconds"] < 1.04 * med]
    rows = []
    for rid, r in clean.groupby("race_id"):
        drivers = sorted(r["driver"].unique())
        lap_ids = sorted(r["lap_number"].unique())
        d_ix = {d: i for i, d in enumerate(drivers)}
        l_ix = {x: i for i, x in enumerate(lap_ids)}
        n = len(r)
        X = np.zeros((n, len(drivers) + len(lap_ids) + 3))
        X[np.arange(n), r["driver"].map(d_ix).values] = 1.0
        X[np.arange(n), len(drivers) + r["lap_number"].map(l_ix).values] = 1.0
        X[:, -3] = (r["compound"] == "SOFT").astype(float).values
        X[:, -2] = (r["compound"] == "MEDIUM").astype(float).values
        X[:, -1] = r["tire_age"].values
        yv = r["lap_time_seconds"].values
        beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
        resid = yv - X @ beta
        sig2 = float(resid @ resid / max(n - np.linalg.matrix_rank(X), 1))
        fe = beta[: len(drivers)]
        cnt = r["driver"].value_counts()
        keep = np.array([cnt.get(d, 0) >= MIN_CLEAN_LAPS for d in drivers])
        fe = fe - fe[keep].mean()
        for d, v, k in zip(drivers, fe, keep):
            if k:
                rows.append({"race_id": rid, "driver": d, "y": float(v), "se2": sig2 / cnt[d], "n_laps": int(cnt[d])})
    out = pd.DataFrame(rows)
    # a corrupt lap set gives a fixed effect tens of seconds off (Italy_2026: -37 s); drop driver-races > 4 s from the
    # race median and re-centre, the remaining ones are real pace (race sd ~1 s/lap)
    med = out.groupby("race_id")["y"].transform("median")
    out = out[(out["y"] - med).abs() < 4.0].copy()
    out["y"] = out["y"] - out.groupby("race_id")["y"].transform("mean")
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------------------------- 1+2 pace
def pace_table(real: Real, lambdas=LAMBDA_GRID) -> pd.DataFrame:
    """One row per (round, driver) with real pace y, its sampling variance and the sim's pace inputs (scale 1, seconds)
    for each shrink lambda: A_<lam> = shrunk quali gap, B_<lam> = rating order quantile-matched onto it."""
    rows = []
    res = real.hist["results"]
    for rnd in real.rounds:
        if rnd < 2:
            continue  # no earlier race, so no walk-forward ratings
        rid = real.round_rid[rnd]
        circuit = res[(res["season"] == 2026) & (res["round"] == rnd)]["circuit_id"].iloc[0]
        track = get_track(G.CIRCUIT_TO_TRACK[circuit])
        ent = G.race_entrants(real.hist, 2026, rnd)
        base = float(track["base_laptime_s"])
        cols = {}
        for lam in lambdas:
            cols[("A", lam)] = G.pace_deltas(real.hist, 2026, rnd, ent, base, 1.0, lam, 1.0)
            cols[("B", lam)] = G.pace_deltas(real.hist, 2026, rnd, ent, base, 1.0, lam, 0.0)
        p = real.pace[real.pace["race_id"] == rid].set_index("driver")
        for i, code in enumerate(ent["code"]):
            if code not in p.index:
                continue
            row = {"round": rnd, "race_id": rid, "code": code, "team_id": ent["team_id"].iloc[i],
                   "y": p.loc[code, "y"], "se2": p.loc[code, "se2"]}
            for (k, lam), v in cols.items():
                row[f"{k}_{lam}"] = float(v[i])
            rows.append(row)
    return pd.DataFrame(rows)


def _demean(df: pd.DataFrame, cols) -> pd.DataFrame:
    return df[cols] - df.groupby("round")[cols].transform("mean")


def fit_pace(tab: pd.DataFrame, train_rounds) -> dict:
    """RACE_PACE_SCALE, DRIVER_SHRINK, QUALI_WEIGHT and the pace-noise components from the training rounds."""
    t = tab[tab["round"].isin(train_rounds)].reset_index(drop=True)
    best = None
    for lam in LAMBDA_GRID:
        dm = _demean(t, ["y", f"A_{lam}", f"B_{lam}"])
        X = dm[[f"A_{lam}", f"B_{lam}"]].to_numpy()
        beta, *_ = np.linalg.lstsq(X, dm["y"].to_numpy(), rcond=None)
        sse = float(((dm["y"].to_numpy() - X @ beta) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, lam, beta)
    sse, lam, beta = best
    cA, cB = float(beta[0]), float(beta[1])
    scale = cA + cB
    if scale <= 0.05:  # degenerate (never seen): fall back to the quali-only slope
        dm = _demean(t, ["y", f"A_{lam}"])
        scale = float((dm["y"] * dm[f"A_{lam}"]).sum() / (dm[f"A_{lam}"] ** 2).sum())
        qw = 1.0
    else:
        qw = float(np.clip(cA / scale, 0.0, 1.0))
        # refit the slope with the clipped weight so scale is consistent with qw
        x = qw * t[f"A_{lam}"] + (1 - qw) * t[f"B_{lam}"]
        t = t.assign(x=x)
        dm = _demean(t, ["y", "x"])
        scale = float((dm["y"] * dm["x"]).sum() / (dm["x"] ** 2).sum())
    t["x"] = qw * t[f"A_{lam}"] + (1 - qw) * t[f"B_{lam}"]
    dm = _demean(t, ["y", "x"])
    t["e"] = dm["y"] - scale * dm["x"]
    # variance components of the residual: teams with two entered drivers in the race
    g = t.groupby(["round", "team_id"])
    two = g["e"].transform("size") == 2
    tt = t[two]
    diff = tt.groupby(["round", "team_id"])["e"].agg(lambda s: s.iloc[0] - s.iloc[1])
    se_pair = tt.groupby(["round", "team_id"])["se2"].sum()
    var_d = max(float((diff ** 2).mean() - se_pair.mean()) / 2.0, 0.0)  # var(e1-e2) = 2 s_d^2 + (se1^2+se2^2)
    tm_mean = tt.groupby(["round", "team_id"])["e"].mean()
    se_mean = tt.groupby(["round", "team_id"])["se2"].mean() / 2.0
    var_t = max(float((tm_mean ** 2).mean() - se_mean.mean()) - var_d / 2.0, 0.0)
    # residual lost to the race demeaning (one df per race): tiny, ignored
    return {
        "race_pace_scale": scale, "driver_shrink": float(lam), "quali_weight": qw,
        "team_pace_sd_s": float(np.sqrt(var_t)), "driver_pace_sd_s": float(np.sqrt(var_d)),
        "n_driver_races": int(len(t)), "resid_sd_s": float(t["e"].std()), "y_sd_s": float(dm["y"].std()),
        "r2": float(1 - (t["e"] ** 2).sum() / (dm["y"] ** 2).sum()),
        "raw_ab": [cA, cB],
    }


# --------------------------------------------------------------------------------------------- 3 SC bunching
def sc_gap_samples(real: Real) -> pd.DataFrame:
    """Per (race, SC episode, lap index k within the episode, k=0 the last green lap): cars' gaps to the car ahead."""
    rows = []
    for rid, r in real.laps.groupby("race_id"):
        fr = r.groupby("lap_number").agg(sc=("has_sc", "mean"), red=("has_red", "mean"))
        laps = fr.index[(fr["sc"] > 0.5) & (fr["red"] < 0.5)].tolist()
        episodes, cur = [], []
        for x in laps:
            if cur and x == cur[-1] + 1:
                cur.append(x)
            else:
                if cur:
                    episodes.append(cur)
                cur = [x]
        if cur:
            episodes.append(cur)
        for e in episodes:
            for k, lap in enumerate([e[0] - 1] + e):
                d = r[(r["lap_number"] == lap) & (r["pit_in_lap"] == 0) & (r["pit_out_lap"] == 0)]
                g = d["ahead_gap"].replace(np.inf, np.nan).dropna()
                g = g[g < 60]
                if len(g) >= 8:
                    rows.append({"race_id": rid, "ep": e[0], "ep_len": len(e), "k": k, "n": len(g), "med": float(g.median())})
    return pd.DataFrame(rows)


def fit_sc(samples: pd.DataFrame, train_races) -> dict:
    s = samples[samples["race_id"].isin(train_races)]
    # floor: median of per-lap median gaps late (k >= 4) in episodes of >= 5 laps (the field has finished closing up)
    late = s[(s["k"] >= 4) & (s["ep_len"] >= 5)]
    floor = float(late["med"].median()) if len(late) >= 3 else float("nan")
    # closure: geometric decay of the median gap between consecutive SC laps from k=2 on, while above 1.5 x the floor
    ratios = []
    for (_, _), g in s.groupby(["race_id", "ep"]):
        m = g.set_index("k")["med"]
        for k in range(2, int(m.index.max())):
            if k in m.index and k + 1 in m.index and m[k] > 1.5 * (floor if np.isfinite(floor) else 0.9) and m[k + 1] > 0:
                ratios.append(m[k + 1] / m[k])
    closure = float(1.0 - np.median(ratios)) if len(ratios) >= 3 else float("nan")
    return {"sc_gap_s": floor, "sc_gap_closure": closure, "n_gap_laps": int(len(late)), "n_ratios": len(ratios)}


# --------------------------------------------------------------------------------------------- 4 lap 1
def lap1_spacing(real: Real, train_races) -> float:
    vals = []
    for rid in train_races:
        r = real.laps[(real.laps["race_id"] == rid) & (real.laps["lap_number"] == 1)].dropna(subset=["position", "gap_leader_s"])
        r = r[r["position"] >= 2]
        if len(r) >= 10:
            vals.append(float((r["gap_leader_s"] / (r["position"] - 1)).median()))
    return float(np.median(vals)) if vals else float("nan")


def fit_held_gap(real: Real, train_races):
    """(median, lognormal sigma) of the end-of-lap gap (s) of an attacker that FAILED to pass and is still behind the
    same car one lap later. sigma = IQR of ln(gap) / 1.349 (robust to the long right tail)."""
    ev = real.events[real.events["race_id"].isin(train_races) & (real.events["passed"] == 0)]
    L = real.laps[["race_id", "driver", "lap_number", "ahead_gap", "road_car_ahead"]].copy()
    L["lap_number"] = L["lap_number"] - 1
    L = L.rename(columns={"ahead_gap": "gap_next", "road_car_ahead": "car_next"})
    m = ev.merge(L, on=["race_id", "driver", "lap_number"], how="left")
    g = m[m["car_next"] == m["defender"]]["gap_next"].replace(np.inf, np.nan).dropna()
    lg = np.log(g[g > 0.02])
    sigma = float((lg.quantile(0.75) - lg.quantile(0.25)) / 1.349)
    return float(g.median()), sigma


# --------------------------------------------------------------------------------------------- overtake / lap1 refit
def fit_interaction_fold(real: Real, train_races) -> dict:
    """T-053 overtake logistic (+ track offsets) and lap-1 table refitted on the training races only."""
    ev = real.events[real.events["race_id"].isin(train_races)].dropna(subset=["pace_delta"])
    tracks = sorted(ev["track_id"].unique())
    t_ix = ev["track_id"].map({t: i for i, t in enumerate(tracks)}).values
    beta = F.fit_logistic(F.design(ev), ev["passed"].values.astype(float), np.ones(len(ev)), t_ix, len(tracks))
    lap1, _ = F.fit_lap1(real.laps[real.laps["race_id"].isin(train_races)])
    return {
        "intercept": float(beta[0]), "pace_delta_coef": float(beta[1]), "tyre_age_delta_coef": float(beta[2]),
        "gap_coef": float(beta[3]), "track_offset": {t: float(o) for t, o in zip(tracks, beta[4:])},
        "lap1": {k: {"mean": v["mean"], "sd": v["sd"]} for k, v in lap1.items()},
    }


# --------------------------------------------------------------------------------------------- 5 strategy mix
def real_strategies(real: Real, rid: str):
    """{driver: [(compound, laps)]} and {driver: [stop laps, green only flag]} from the lap table."""
    r = real.laps[real.laps["race_id"] == rid]
    out, stops = {}, {}
    for d, g in r.sort_values("lap_number").groupby("driver"):
        g = g.reset_index(drop=True)
        comp = g["compound"].fillna("?").to_numpy()
        stint = g["tyre_stint"].to_numpy()
        cuts = [i for i in range(1, len(g)) if stint[i] != stint[i - 1]]
        edges = [0] + cuts + [len(g)]
        out[d] = [(comp[edges[i]], edges[i + 1] - edges[i]) for i in range(len(edges) - 1)]
        sc = (g["has_sc"] | g["has_vsc"] | g["has_red"]).to_numpy(bool)
        stops[d] = [(c, bool(sc[c - 1] or sc[max(c - 2, 0)])) for c in cuts]
    return out, stops


def plan_distance(plan, strat) -> float:
    """Distance between an optimizer plan [(compound, laps)] and a real strategy (same format)."""
    ns = abs(len(plan) - len(strat))
    pe = np.cumsum([n for _, n in plan])[:-1]
    se = np.cumsum([n for _, n in strat])[:-1]
    m = min(len(pe), len(se))
    lap_d = float(np.abs(pe[:m] - se[:m]).sum())
    comp_d = sum(1 for (c1, _), (c2, _) in zip(plan, strat) if c1 != c2)
    return 15.0 * ns + lap_d + 5.0 * comp_d


def strategy_mix_stats(real: Real, rid: str, plans) -> dict:
    """Per race: counts of drivers nearest to each plan and the matched green-flag stop-lap residuals."""
    strats, stops = real_strategies(real, rid)
    plans_t = [[(s.compound, s.laps) for s in p] for p in plans]
    counts = np.zeros(len(plans_t))
    resid = []
    for d, st in strats.items():
        if sum(n for _, n in st) < 10 or len(st) > 4:
            continue
        dist = [plan_distance(p, st) for p in plans_t]
        k = int(np.argmin(dist))
        counts[k] += 1
        pe = np.cumsum([n for _, n in plans_t[k]])[:-1]
        se = np.cumsum([n for _, n in st])[:-1]
        for j in range(min(len(pe), len(se))):
            if not stops[d][j][1]:  # green-flag stop only
                resid.append(float(se[j] - pe[j]))
    return {"counts": counts.tolist(), "resid": resid}


def fit_strategy_mix(stats: dict, train_races) -> dict:
    cnt = np.sum([stats[r]["counts"] for r in train_races if r in stats], axis=0)
    resid = np.concatenate([stats[r]["resid"] for r in train_races if r in stats])
    w = cnt / cnt.sum()
    sd = float(np.std(resid)) if len(resid) > 5 else float("nan")
    # integer J such that uniform[-J, J] has this sd: sd = sqrt(((2J+1)^2 - 1) / 12) = sqrt(J(J+1)/3)
    J = int(np.clip(np.round((-1 + np.sqrt(1 + 12 * sd ** 2)) / 2), 0, 12)) if np.isfinite(sd) else None
    return {"plan_weights": [float(x) for x in w], "pit_jitter_laps": J, "resid_sd_laps": sd, "n_drivers": int(cnt.sum()),
            "n_resid": int(len(resid))}


# --------------------------------------------------------------------------------------------- fold track params
import build_track_params as B  # noqa: E402

PREFIX_OF_TRACK = {}
for _cid, (_circuit, _prefix) in B.TRACKS.items():
    PREFIX_OF_TRACK["las_vegas" if _prefix == "LasVegas" else _prefix.lower()] = _prefix

_ROUND_OF = {}


def round_of_track(real: Real) -> dict:
    """{track_id: round} for the 2026 races with lap data."""
    if not _ROUND_OF:
        res = real.hist["results"]
        for rnd in real.rounds:
            circuit = res[(res["season"] == 2026) & (res["round"] == rnd)]["circuit_id"].iloc[0]
            _ROUND_OF[G.CIRCUIT_TO_TRACK[circuit]] = rnd
    return _ROUND_OF


def q1_best_s(hist: dict, rnd: int) -> float:
    q = hist["qualifying"]
    q = q[(q["season"] == 2026) & (q["round"] == rnd)]
    t = pd.to_numeric(q["q1"].map(G._lap_seconds), errors="coerce")
    return float(t.min())


def rebuild_track(track_id: str, rnd: int, real: Real, model: dict) -> dict:
    """
    track_params entry for a race WITHOUT that race's own data (T-060, the validation leak fix).
      total_laps: the race distance (known before the race), kept from track_params.
      base_laptime_s: ratio x this weekend's best Q1 time, ratio = median(base lap / best Q1) over the OTHER 2026 races
        (a pole-to-race-pace conversion known before the start; mean abs error vs the leaky value 0.87 s = 1.0%; the
        same-track 2025 race was not used for the base lap because 2025 cars are a different formula).
      pit_loss_s / noise / stints: the same-track 2025 race when it exists, is dry and has >= 5 clean stops (the pit lane
        is a track property and this is what production does for an unraced track), else the median of the other
        2026 tracks.
    """
    full = get_track(track_id)
    rot = round_of_track(real)
    others = [t for t in G_tracks() if t["track_id"] != track_id and t.get("already_raced_2026") and t["track_id"] in rot]
    ratios = [t["base_laptime_s"] / q1_best_s(real.hist, rot[t["track_id"]]) for t in others]
    out = {k: full[k] for k in ("track_id", "name", "circuit", "total_laps", "compounds", "on_2026_calendar")}
    out["base_laptime_s"] = round(float(np.median(ratios)) * q1_best_s(real.hist, rnd), 2)
    key = f"{PREFIX_OF_TRACK[track_id]}_2025"
    loaded = None if key in B.EXCLUDE_RACES else B.load_race(key)
    use25 = loaded is not None and B.wet_fraction(loaded[0]) <= B.WET_EXCLUDE_FRAC
    if use25:
        d = B.pit_durations(loaded[1])
        if len(d) >= 5:
            out["pit_loss_s"] = round(float(np.median(d)), 2)
            d60 = d[d <= 60]
            out["pit_loss_noise_std_s"] = round(float(d60.std(ddof=1)), 2) if len(d60) > 2 else float("nan")
        sl = B.stint_lengths_flagged(loaded[0])
        out["max_stint_laps"] = {c: int(max(x for x, _ in sl[c] if x >= 5)) for c in B.DRY if any(x >= 5 for x, _ in sl[c])}
        out["typical_stint_laps"] = {}
        for c in out["max_stint_laps"]:
            v = [x for x, red in sl[c] if x >= 5 and not red]
            out["typical_stint_laps"][c] = int(round(float(np.percentile(v, 90)))) if len(v) >= 5 else None
    out.setdefault("max_stint_laps", {})
    out.setdefault("typical_stint_laps", {})
    for c in B.DRY:
        if c not in out["max_stint_laps"]:
            mv = [t["max_stint_laps"][c] for t in others if c in t["max_stint_laps"]]
            if mv:
                out["max_stint_laps"][c] = int(np.median(mv))
                ty = [t["typical_stint_laps"].get(c) for t in others if t.get("typical_stint_laps", {}).get(c)]
                out["typical_stint_laps"][c] = int(np.median(ty)) if ty else None
    if "pit_loss_s" not in out:
        out["pit_loss_s"] = round(float(np.median([t["pit_loss_s"] for t in others])), 2)
        out["pit_loss_noise_std_s"] = round(float(np.median([t["pit_loss_noise_std_s"] for t in others])), 2)
    out["source_note"] = f"T-060 fold track: no {track_id} 2026 race data used"
    return out


def G_tracks():
    from app.core.track_params import load_tracks

    return load_tracks()


# --------------------------------------------------------------------------------------------- study (validation)
import dataclasses  # noqa: E402

from scipy.stats import spearmanr  # noqa: E402

from app.core.race_grid_sim import (  # noqa: E402
    GridCalibration,
    load_interaction_params,
    rank_by_points,
    simulate_grid,
)
from app.core.safety_cars import load_params as load_sc  # noqa: E402

WF_PATH = ROOT / "data" / "jolpica" / "predictions" / "walk_forward_2026_drivers.csv"
PASS_SCALES = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)
PASS_SHIFTS = (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5)
# the coarse 2-D grid (evidence that the scale is not identified by pass counts) plus a fine line at scale 1 (the shipped
# choice: the T-053 pace coefficient is kept and only the intercept is re-fitted)
PASS_CONFIGS = [(a, b) for a in PASS_SCALES for b in PASS_SHIFTS] + [(1.0, round(-1.2 + 0.05 * k, 2)) for k in range(11)]
PASS_CONFIGS = sorted(set(PASS_CONFIGS))


def round_to_file() -> dict:
    out = {}
    for p in (ROOT / "data" / "processed").glob("*_race_info.json"):
        info = json.loads(p.read_text())
        if info.get("season") == 2026 and info.get("round"):
            out[int(info["round"])] = p.name.replace("_race_info.json", "_laps.csv")
    return out


def score_order(pred_rank, actual_pos, classified) -> dict:
    rho = spearmanr(pred_rank, actual_pos).statistic
    c = classified.astype(bool)
    rho_c = spearmanr(pred_rank[c], actual_pos[c]).statistic if c.sum() > 3 else np.nan
    return {
        "spearman": float(rho), "spearman_classified": float(rho_c),
        "top3_hits": len(set(np.argsort(pred_rank)[:3]) & set(np.argsort(actual_pos)[:3])) / 3.0,
        "winner": float(np.argmin(pred_rank) == np.argmin(actual_pos)),
    }


class Study:
    """Per-round fold contexts (tyre model, track entry) built without the held-out race."""

    def __init__(self, real: Real, eval_rounds, verbose=True):
        self.real = real
        self.hist = real.hist
        self.eval_rounds = list(eval_rounds)
        laps_all = tm.load_all_laps()
        files = round_to_file()
        res = self.hist["results"]
        self.ctx = {}
        for rnd in sorted(set(self.eval_rounds) | {1}):
            r = res[(res["season"] == 2026) & (res["round"] == rnd)]
            tid = G.CIRCUIT_TO_TRACK[r["circuit_id"].iloc[0]]
            f = files.get(rnd)
            lt = laps_all if f is None else laps_all[~laps_all["race_file"].str.replace("\\", "/").str.endswith(f)]
            model = tm.build_tire_model(laps=lt)
            track = rebuild_track(tid, rnd, real, model)
            self.ctx[rnd] = {"model": model, "track": track, "track_id": tid, "rid": real.round_rid.get(rnd)}
            if verbose:
                print(f"  fold ctx R{rnd} {tid}", flush=True)
        self.sc_samples = sc_gap_samples(real)
        self.pace_tab = pace_table(real)
        self.strat_stats = {}
        for rnd, c in self.ctx.items():
            if c["rid"] is None:
                continue
            plans = G.top_plans(c["track"], c["model"], int(c["track"]["total_laps"]))
            self.strat_stats[c["rid"]] = strategy_mix_stats(real, c["rid"], plans)
        self.passes = real.passes_per_race()  # by round
        self.wf = pd.read_csv(WF_PATH)

    # ---- constants fitted on a training set of rounds
    def fit_all(self, train_rounds) -> dict:
        real = self.real
        train_rids = [real.round_rid[r] for r in train_rounds if r in real.round_rid]
        return {
            "pace": fit_pace(self.pace_tab, [r for r in train_rounds if r >= 2]),
            "sc": fit_sc(self.sc_samples, train_rids),
            "lap1_spacing_s": lap1_spacing(real, train_rids),
            "held_gap_s": fit_held_gap(real, train_rids)[0],
            "held_gap_sigma": fit_held_gap(real, train_rids)[1],
            "strategy_mix": fit_strategy_mix(self.strat_stats, train_rids),
            "interaction": fit_interaction_fold(real, train_rids),
            "train_rounds": list(train_rounds),
        }

    def make_cal(self, fit: dict, scale=1.0, shift=0.0, use_held=True) -> GridCalibration:
        p, sc, mix = fit["pace"], fit["sc"], fit["strategy_mix"]
        d = GridCalibration()
        return GridCalibration(
            team_pace_sd_s=p["team_pace_sd_s"], driver_pace_sd_s=p["driver_pace_sd_s"],
            sc_gap_s=sc["sc_gap_s"] if np.isfinite(sc["sc_gap_s"]) else d.sc_gap_s,
            sc_gap_closure=sc["sc_gap_closure"] if np.isfinite(sc["sc_gap_closure"]) else d.sc_gap_closure,
            lap1_spacing_s=fit["lap1_spacing_s"], ov_pace_scale=scale, ov_intercept_shift=shift,
            held_gap_s=fit["held_gap_s"] if use_held else None, held_gap_sigma=fit["held_gap_sigma"],
            race_pace_scale=p["race_pace_scale"], driver_shrink=p["driver_shrink"], quali_weight=p["quali_weight"],
            plan_weights=tuple(mix["plan_weights"]),
            pit_jitter_laps=mix["pit_jitter_laps"] if mix["pit_jitter_laps"] is not None else d.pit_jitter_laps,
            source="fitted (T-060 study)",
        )

    def make_ip(self, fit: dict):
        ip = load_interaction_params()
        i = fit["interaction"]
        return dataclasses.replace(
            ip, ov_intercept=i["intercept"], ov_pace_coef=i["pace_delta_coef"], ov_tyre_age_coef=i["tyre_age_delta_coef"],
            ov_gap_coef=i["gap_coef"], ov_track_offset=i["track_offset"], lap1_bins=i["lap1"],
        )

    def inputs(self, rnd, cal: GridCalibration, leaky: bool):
        c = self.ctx[rnd]
        kw = dict(rnd=rnd, seed=rnd, hist=self.hist, race_pace_scale=cal.race_pace_scale, quali_weight=cal.quali_weight,
                  driver_shrink=cal.driver_shrink, plan_weights=cal.plan_weights, pit_jitter=cal.pit_jitter_laps)
        if leaky:  # T-034 as committed: track_params.json built with the held-out race
            return G.build_inputs(c["track_id"], c["model"], **kw)
        return G.build_inputs(c["track_id"], c["model"], track=c["track"], **kw)

    def run_race(self, rnd, cal: GridCalibration, ip, n_sims, leaky=False, gi=None):
        c = self.ctx[rnd]
        gi = gi or self.inputs(rnd, cal, leaky)
        sim = simulate_grid(
            gi.cars, gi.track["base_laptime_s"], c["model"], gi.total_laps, n_sims=n_sims, seed=rnd,
            pit_loss_s=gi.track["pit_loss_s"], interaction=ip, safety_cars=load_sc(), track_id=c["track_id"], calibration=cal,
        )
        return gi, sim

    @staticmethod
    def position_change_stats(sim):
        """Mean over runs of the per-run (sd, mean abs) of grid - finishing position among cars that did not retire."""
        pos, ret, grid = sim.positions.astype(float), sim.retired, sim.grid.astype(float)
        ch = grid[None, :] - pos
        n_ok = (~ret).sum(axis=1).astype(float)
        mean = np.where(~ret, ch, 0.0).sum(axis=1) / n_ok
        var = (np.where(~ret, (ch - mean[:, None]) ** 2, 0.0).sum(axis=1)) / np.maximum(n_ok - 1, 1)
        mac = np.where(~ret, np.abs(ch), 0.0).sum(axis=1) / n_ok
        return float(np.sqrt(var).mean()), float(mac.mean())

    def score(self, rnd, gi, sim) -> dict:
        s = sim.summary()
        codes = [x["code"] for x in s]
        pr = rank_by_points([x["exp_points"] for x in s], [x["mean_position"] for x in s])
        res = self.hist["results"]
        er = res[(res["season"] == 2026) & (res["round"] == rnd)].set_index("code").loc[codes]
        actual = er["position"].to_numpy(dtype=float)
        cls = er["classified"].to_numpy()
        grid = np.array([x["grid"] for x in s])
        row = {"round": rnd, "track": self.ctx[rnd]["track_id"], "n": len(codes), "sim_passes": float(sim.n_overtakes.mean())}
        for name, rk in (("sim", pr), ("grid", grid.astype(float))):
            for k, v in score_order(np.asarray(rk, float), actual, cls).items():
                row[f"{name}_{k}"] = v
        w = self.wf[self.wf["round"] == rnd].set_index("code")
        winner, is_pod = int(np.argmin(actual)), (actual <= 3).astype(float)

        def prob_scores(name, pw, pp):
            row[f"{name}_win_ll"] = float(-np.log(max(float(pw[winner]), 1e-3)))
            row[f"{name}_pod_brier"] = float(np.mean((np.asarray(pp) - is_pod) ** 2))

        sp_win = np.array([x["p_win"] for x in s])
        sp_pod = np.array([x["p_podium"] for x in s])
        prob_scores("sim", sp_win, sp_pod)
        if len(w) == len(codes) and set(w.index) == set(codes):
            for k, v in score_order(w.loc[codes, "pred_rank"].to_numpy(float), actual, cls).items():
                row[f"pl_{k}"] = v
            wl = w.loc[codes]
            prob_scores("pl", wl["p_win"].to_numpy(float), wl["p_podium"].to_numpy(float))
            # blends of the PL predictor and the sim (pre-specified 50/50; 25/75 reported as sensitivity)
            sim_pts = np.array([x["exp_points"] for x in s])
            sim_pos = np.array([x["mean_position"] for x in s])
            for tag, wt in (("blend50", 0.5), ("blend25", 0.25), ("blend75", 0.75)):  # wt = weight on the sim
                pts = wt * sim_pts + (1 - wt) * wl["exp_pts"].to_numpy(float)
                pos = wt * sim_pos + (1 - wt) * wl["exp_pos"].to_numpy(float)
                for k, v in score_order(rank_by_points(pts, pos).astype(float), actual, cls).items():
                    row[f"{tag}_{k}"] = v
                prob_scores(tag, wt * sp_win + (1 - wt) * wl["p_win"].to_numpy(float),
                            wt * sp_pod + (1 - wt) * wl["p_podium"].to_numpy(float))
        row["sim_pc_sd"], row["sim_pc_mac"] = self.position_change_stats(sim)
        row["real_pc_sd"], row["real_pc_mac"] = self.real.position_change(rnd, gi.ent)
        row["real_passes"] = float(self.passes.get(rnd, np.nan))
        return row


def summarise_rows(df: pd.DataFrame) -> dict:
    out = {"n_races": int(len(df))}

    def se(x):
        return float(x.std(ddof=1) / np.sqrt(len(x)))

    for m in ("sim", "grid", "pl", "blend50", "blend25", "blend75"):
        c = f"{m}_spearman"
        if c in df:
            d = df[df[c].notna()]
            out[m] = {
                "spearman": float(d[c].mean()), "spearman_se": se(d[c]),
                "spearman_classified": float(d[f"{m}_spearman_classified"].mean()),
                "top3_hit_rate": float(d[f"{m}_top3_hits"].mean()), "top3_se": se(d[f"{m}_top3_hits"]),
                "winner_hit_rate": float(d[f"{m}_winner"].mean()), "winner_se": se(d[f"{m}_winner"]),
                "n": int(len(d)),
            }
    for m in ("sim", "pl", "blend50", "blend25", "blend75"):
        if f"{m}_win_ll" in df:
            d = df[df[f"{m}_win_ll"].notna() & df["pl_win_ll"].notna()] if "pl_win_ll" in df else df
            out.setdefault("prob_scores", {})[m] = {
                "win_logloss": float(d[f"{m}_win_ll"].mean()), "win_logloss_se": se(d[f"{m}_win_ll"]),
                "podium_brier": float(d[f"{m}_pod_brier"].mean()), "podium_brier_se": se(d[f"{m}_pod_brier"]), "n": int(len(d))}
    for a, b in (("sim", "grid"), ("sim", "pl"), ("blend50", "pl"), ("blend50", "grid")):
        if f"{a}_spearman" in df and f"{b}_spearman" in df:
            d = (df[f"{a}_spearman"] - df[f"{b}_spearman"]).dropna()
            out[f"{a}_minus_{b}_spearman"] = {"mean": float(d.mean()), "se": se(d), "races_better": int((d > 0).sum()), "n": int(len(d))}
    pr = df.dropna(subset=["real_passes"])
    if len(pr):
        out["passes"] = {"sim_mean": float(pr["sim_passes"].mean()), "real_mean": float(pr["real_passes"].mean()),
                         "corr_per_race": float(np.corrcoef(pr["sim_passes"], pr["real_passes"])[0, 1]) if len(pr) > 3 else None}
    out["position_change"] = {"sim_sd": float(df["sim_pc_sd"].mean()), "real_sd": float(df["real_pc_sd"].mean()),
                              "sim_mac": float(df["sim_pc_mac"].mean()), "real_mac": float(df["real_pc_mac"].mean())}
    return out


# --------------------------------------------------------------------------------------------- 2025 out-of-sample season
EXCLUDED_2025_TRACKS = {"australia", "belgium", "britain"}  # wet 2025 races (tire_model.EXCLUDED_2025_RACES): a dry tyre model does not apply


def evaluate_2025(st: Study, table: pd.DataFrame, n_sims: int = 2000, verbose=True) -> dict:
    """
    Second season, fully out of sample for the sim: every constant (tyre model, interaction parameters, calibration,
    overtake shift) is the all-2026 fit, so no 2025 race was ever used. Tracks whose track_params entry came from the
    2025 race itself (Brazil, Las Vegas, Qatar, Abu Dhabi) are skipped (pit loss / stint caps would leak), wet 2025
    races are skipped. The base lap time is ratio x this weekend's best Q1 (2025 cars), pit loss and stints are the
    2026 same-track values. The Plackett-Luce comparison is P.predict_race(mode='grid') for the round, which only sees
    earlier races; NOTE its config was tuned on the 2025 walk-forward (predictor.PredictorConfig), so 2025 favours PL.
    """
    from app.core import predictor as P

    hist = st.hist
    fit = st.fit_all(st.real.rounds)
    cfg = choose_config(table)
    cal = st.make_cal(fit, cfg["scale"], cfg["shift"])
    ip = st.make_ip(fit)
    model = tm.build_tire_model()  # all 2026 laps, no 2025 race
    rot = round_of_track(st.real)
    res = hist["results"]
    sched = hist["schedule"]
    s25 = sched[sched["season"] == 2025].sort_values("round")
    rows = []
    for _, r in s25.iterrows():
        rnd, cid = int(r["round"]), r["circuit_id"]
        tid = G.CIRCUIT_TO_TRACK.get(cid)
        if tid is None or tid in EXCLUDED_2025_TRACKS or rnd < 2:
            continue
        full = get_track(tid)
        if full is None or full.get("source_season") == 2025 and not full.get("already_raced_2026"):
            continue
        q = hist["qualifying"]
        if q[(q["season"] == 2025) & (q["round"] == rnd)].empty or res[(res["season"] == 2025) & (res["round"] == rnd)].empty:
            continue
        # base lap ratio from the 2026 races
        others = [t for t in G_tracks() if t.get("already_raced_2026") and t["track_id"] in rot]
        ratio = float(np.median([t["base_laptime_s"] / q1_best_s(hist, rot[t["track_id"]]) for t in others]))
        q1b = float(pd.to_numeric(q[(q["season"] == 2025) & (q["round"] == rnd)]["q1"].map(G._lap_seconds), errors="coerce").min())
        track = dict(full)
        track["base_laptime_s"] = round(ratio * q1b, 2)
        try:
            gi = G.build_inputs(tid, model, season=2025, rnd=rnd, seed=rnd, hist=hist, track=track,
                                race_pace_scale=cal.race_pace_scale, quali_weight=cal.quali_weight,
                                driver_shrink=cal.driver_shrink, plan_weights=cal.plan_weights, pit_jitter=cal.pit_jitter_laps)
            sim = simulate_grid(gi.cars, track["base_laptime_s"], model, gi.total_laps, n_sims=n_sims, seed=rnd,
                                pit_loss_s=track["pit_loss_s"], interaction=ip, safety_cars=load_sc(), track_id=tid, calibration=cal)
        except Exception as exc:  # a race with a changed entry list / missing data: skip, say so
            if verbose:
                print(f"  2025 R{rnd} {tid} skipped: {exc!r}")
            continue
        s = sim.summary()
        codes = [x["code"] for x in s]
        er = res[(res["season"] == 2025) & (res["round"] == rnd)].set_index("code")
        if not set(codes) <= set(er.index):
            continue
        er = er.loc[codes]
        actual = er["position"].to_numpy(dtype=float)
        cls = er["classified"].to_numpy()
        pr = rank_by_points([x["exp_points"] for x in s], [x["mean_position"] for x in s])
        grid = np.array([x["grid"] for x in s], dtype=float)
        row = {"round": rnd, "track": tid, "n": len(codes), "sim_passes": float(sim.n_overtakes.mean())}
        for name, rk in (("sim", pr), ("grid", grid)):
            for k, v in score_order(np.asarray(rk, float), actual, cls).items():
                row[f"{name}_{k}"] = v
        try:
            ent = gi.ent
            pl = P.predict_race(hist, 2025, rnd, "grid", entry_list=ent[["driver_id", "code", "team_id", "team"]],
                                grid=ent["grid"].to_numpy(dtype=float))
            pl = pl.set_index("code").loc[codes]
            plr = P.rank_by_points(pl["exp_pts"].to_numpy(float), pl["exp_pos"].to_numpy(float))
            for k, v in score_order(np.asarray(plr, float), actual, cls).items():
                row[f"pl_{k}"] = v
        except Exception as exc:
            if verbose:
                print(f"  2025 R{rnd} PL failed: {exc!r}")
        row["sim_pc_sd"], row["sim_pc_mac"] = Study.position_change_stats(sim)
        # real position change among classified
        e = gi.ent.set_index("code")
        d = pd.DataFrame({"grid": e["grid"], "pos": er["position"], "cls": er["classified"]}).dropna()
        d = d[d["cls"].astype(bool)]
        ch = (d["grid"] - d["pos"]).to_numpy(float)
        row["real_pc_sd"], row["real_pc_mac"] = float(ch.std(ddof=1)), float(np.abs(ch).mean())
        row["real_passes"] = np.nan
        rows.append(row)
        if verbose:
            print(f"2025 R{rnd:>2} {tid:<12} sim {row['sim_spearman']:.2f} grid {row['grid_spearman']:.2f} pl {row.get('pl_spearman', float('nan')):.2f} "
                  f"passes {row['sim_passes']:.0f}", flush=True)
    return {"rows": pd.DataFrame(rows)}


# --------------------------------------------------------------------------------------------- pass calibration
def selection_table(st: Study, n_sims: int = 600, rounds=None, verbose=True) -> pd.DataFrame:
    """
    Sim passes / position-change stats per (round, pace_delta_scale, intercept_shift), run with the constants fitted on ALL
    races (the constants are shared across the grid so this is 19 x |grid| sims, not 19 x 19 x |grid|). The fold choice
    below leaves the held-out race out of the TARGETS. Residual leak: the all-race pace/noise/SC constants used here
    include the held-out race at weight 1/19; the choice is then tested with fold-fitted constants (evaluate_modes).
    """
    rounds = rounds or [r for r in st.eval_rounds if r in st.real.round_rid]
    fit = st.fit_all(st.real.rounds)
    ip = st.make_ip(fit)
    rows = []
    for rnd in rounds:
        gi = st.inputs(rnd, st.make_cal(fit), leaky=False)
        for sc, sh in PASS_CONFIGS:
            _, sim = st.run_race(rnd, st.make_cal(fit, sc, sh), ip, n_sims, gi=gi)
            sd, mac = st.position_change_stats(sim)
            rpc = st.real.position_change(rnd, gi.ent)
            rows.append({"round": rnd, "scale": sc, "shift": sh, "passes": float(sim.n_overtakes.mean()),
                         "pc_sd": sd, "pc_mac": mac, "real_passes": float(st.passes[rnd]), "real_pc_sd": rpc[0], "real_pc_mac": rpc[1]})
        if verbose:
            print(f"  selection sims R{rnd}", flush=True)
    return pd.DataFrame(rows)


def choose_config(table: pd.DataFrame, exclude_round=None, fix_scale=1.0) -> dict:
    """(scale, shift) minimising the squared relative error of mean passes/race, mean position-change sd and mean
    absolute position change vs their real means, over the races other than `exclude_round`. fix_scale=1.0 (default)
    keeps the T-053 pace coefficient and fits only the intercept shift: the coarse 2-D table shows pass count and
    position-change spread lie on a ridge in (scale, shift) (scale 0 .. 1.25 all reach the targets with a matching
    shift), so the scale is not identified by these statistics. fix_scale=None searches both."""
    t = table if exclude_round is None else table[table["round"] != exclude_round]
    if fix_scale is not None:
        t = t[t["scale"] == fix_scale]
    g = t.groupby(["scale", "shift"]).agg(p=("passes", "mean"), rp=("real_passes", "mean"), s=("pc_sd", "mean"),
                                          rs=("real_pc_sd", "mean"), m=("pc_mac", "mean"), rm=("real_pc_mac", "mean")).reset_index()
    g["loss"] = ((g["p"] - g["rp"]) / g["rp"]) ** 2 + ((g["s"] - g["rs"]) / g["rs"]) ** 2 + ((g["m"] - g["rm"]) / g["rm"]) ** 2
    b = g.sort_values("loss").iloc[0]
    return {"scale": float(b["scale"]), "shift": float(b["shift"]), "loss": float(b["loss"]), "sim_passes": float(b["p"]),
            "real_passes": float(b["rp"]), "sim_pc_sd": float(b["s"]), "real_pc_sd": float(b["rs"])}


def evaluate_modes(st: Study, table: pd.DataFrame, n_sims: int = 2000, verbose=True) -> dict:
    """Leave-one-race-out scorecard of the sim in four configurations (see the handoff for the reading):
    t034_leaky      T-034 as committed (placeholders, track_params and race_interaction.json built with the held-out race)
    t034_leak_fixed same placeholders, fold track entry + fold-refitted interaction params (held-out track offset = 0)
    fitted          fold-fitted data constants (pace, noise, SC, lap-1, strategy mix, held gap), no overtake scale/shift
    calibrated      fitted + the fold-chosen overtake intercept shift (pace scale kept at 1)
    calibrated_2d   fitted + fold-chosen (pace scale, intercept shift) searched jointly (sensitivity)"""
    out = {m: [] for m in ("t034_leaky", "t034_leak_fixed", "fitted", "calibrated", "calibrated_2d")}
    chosen = {}
    for rnd in st.eval_rounds:
        tr = [r for r in st.real.rounds if r != rnd]
        f = st.fit_all(tr)
        ip = st.make_ip(f)
        cfg = choose_config(table, exclude_round=rnd)
        cfg2 = choose_config(table, exclude_round=rnd, fix_scale=None)
        chosen[rnd] = {"scale1": cfg, "free": cfg2}
        runs = {
            "t034_leaky": (GridCalibration(), load_interaction_params(), True),
            "t034_leak_fixed": (GridCalibration(), ip, False),
            "fitted": (st.make_cal(f), ip, False),
            "calibrated": (st.make_cal(f, cfg["scale"], cfg["shift"]), ip, False),
            "calibrated_2d": (st.make_cal(f, cfg2["scale"], cfg2["shift"]), ip, False),
        }
        for name, (cal, ipx, leaky) in runs.items():
            gi, sim = st.run_race(rnd, cal, ipx, n_sims, leaky=leaky)
            out[name].append(st.score(rnd, gi, sim))
        if verbose:
            r = out["calibrated"][-1]
            print(f"R{rnd:>2} {r['track']:<12} rho cal {r['sim_spearman']:.2f} fit {out['fitted'][-1]['sim_spearman']:.2f} "
                  f"t034 {out['t034_leaky'][-1]['sim_spearman']:.2f} grid {r['grid_spearman']:.2f} pl {r.get('pl_spearman', float('nan')):.2f} | "
                  f"passes {r['sim_passes']:.0f} (real {r['real_passes']:.0f}) cfg {cfg['scale']}/{cfg['shift']}", flush=True)
    return {"rows": {k: pd.DataFrame(v) for k, v in out.items()}, "chosen": chosen}


def build_output(st: Study, table: pd.DataFrame) -> dict:
    """The shipped calibration file: constants fitted on all races + the pass calibration chosen on all races."""
    fit = st.fit_all(st.real.rounds)
    cfg = choose_config(table)
    p, sc, mix = fit["pace"], fit["sc"], fit["strategy_mix"]
    return {
        "version": 1,
        "fitted_on": st.real.races,
        "pace": {"race_pace_scale": round(p["race_pace_scale"], 4), "driver_shrink": p["driver_shrink"],
                 "quali_weight": round(p["quali_weight"], 4), "team_pace_sd_s": round(p["team_pace_sd_s"], 4),
                 "driver_pace_sd_s": round(p["driver_pace_sd_s"], 4), "r2": round(p["r2"], 4),
                 "resid_sd_s": round(p["resid_sd_s"], 4), "n_driver_races": p["n_driver_races"],
                 "raw_ab": [round(x, 4) for x in p["raw_ab"]]},
        "sc": {"gap_s": sc["sc_gap_s"], "closure": round(sc["sc_gap_closure"], 4), "n_gap_laps": sc["n_gap_laps"], "n_ratios": sc["n_ratios"]},
        "lap1_spacing_s": fit["lap1_spacing_s"],
        "strategy_mix": {"plan_weights": [round(x, 4) for x in mix["plan_weights"]], "pit_jitter_laps": mix["pit_jitter_laps"],
                         "resid_sd_laps": round(mix["resid_sd_laps"], 3), "n_drivers": mix["n_drivers"]},
        "overtake_calibration": {"pace_delta_scale": cfg["scale"], "intercept_shift": cfg["shift"], "held_gap_s": fit["held_gap_s"], "held_gap_sigma": round(fit["held_gap_sigma"], 4),
                                 "target_passes_per_race": round(cfg["real_passes"], 2), "sim_passes_per_race": round(cfg["sim_passes"], 2),
                                 "target_position_change_sd": round(cfg["real_pc_sd"], 3), "sim_position_change_sd": round(cfg["sim_pc_sd"], 3)},
        "notes": (
            "Fitted 2026-11-16 by calibrate_grid_sim.py (T-060) from the 19 dry 2026 races. pace.* regress real clean-air race "
            "pace on the sim's own pace inputs (scale 1): race_pace_scale is the slope, quali_weight the share of the Q1 "
            "gap vs the quantile-matched rating order, driver_shrink the pull toward the team mean, *_pace_sd_s the "
            "variance components of the residual net of the pace-estimate sampling variance. sc.gap_s / closure: bunching "
            "behind the SC (floor and per-lap decay of the median car-to-car gap). lap1_spacing_s: slope of gap-to-leader "
            "at the end of lap 1 per position. strategy_mix: share of real strategies nearest to the optimizer's top-3 plans "
            "and the half-width of the per-stop jitter. overtake_calibration: pace_delta_scale multiplies the pace delta in "
            "the overtake logit, intercept_shift is added to the intercept, held_gap_s is the gap a car is held at after "
            "a failed pass (real median); scale/shift chosen to match real passes per race (T-053 definition, all "
            "events) and the spread of grid->finish position change."
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--n-sims", type=int, default=600)
    ap.add_argument("--table", default=None, help="reuse a selection table csv instead of re-running the sims")
    a = ap.parse_args()
    t0 = time.time()
    real = Real()
    st = Study(real, range(2, 21), verbose=False)
    print(f"{len(real.races)} races, study built in {time.time() - t0:.0f}s")
    table = pd.read_csv(a.table) if a.table else selection_table(st, a.n_sims)
    if not a.table:
        TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(TABLE_PATH, index=False)
    out = build_output(st, table)
    print(json.dumps({k: v for k, v in out.items() if k != "notes"}, indent=1))
    if not a.no_write:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
