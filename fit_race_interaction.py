"""
Fit the race-interaction parameters of the full-grid simulation (T-053, Phase 1 of team/ROADMAP.md) from real 2026
lap data and write data/sim_params/race_interaction.json.

Run this from project root with: python fit_race_interaction.py
(add --no-write to only print; --quick halves the bootstrap counts).

Data: every dry 2026 race in data/processed/*_laps.csv plus data/staging/*_2026_laps.csv for races not yet in
processed (Malaysia_2026). Wet races are dropped with tire_model.drop_wet_races (same rule as the tyre fit). The laps
CSVs already carry the race-order position at the end of each lap and the on-road gap to the car that crossed the line
before (road_gap_ahead_s, road_car_ahead, road_ahead_lap_diff). All gaps below are those end-of-lap timing-line gaps, so
they are coarse (one sample per lap). 2025 staging races are used ONLY as a down-weighted prior and ONLY if they beat a
2026-only model in leave-one-race-out (see section 2, "2025 prior test").

What is fitted and how (the long version of the 2026-11-16 decisions, since several choices are not obvious):

1. Dirty air. Sample = tire_model.clean_laps() (green, dry, non-pit, not lap 1, not restart lap) with a known on-road
   gap to a car on the same lap (a lap-down car ahead is dropped, a leader gets "clean air"). Lap time is regressed on
   stint fixed effects (kills driver pace, compound, fuel start) + race-lap fixed effects (kills the common fuel burn,
   track evolution and weather drift: each lap is compared with the field on the SAME lap) + tyre age + gap-bin
   dummies. FEs are nested in a race, so they are removed per race (alternating projections) and the remaining small
   regression is bootstrapped by RACE for confidence intervals. Extra wear = coefficient of (in-window x tyre age) over
   the pooled age slope. Caveat: the gap is measured at the END of the lap, so a car that is catching a slow car is
   "dirty" partly because it was already slow; the stint FE removes the level but not that reverse causation, so the loss
   is an upper-ish estimate.
2. Overtake. One observation per (attacker A, lap n) where A's on-road car ahead at the end of lap n is D, same lap, gap
   <= OVERTAKE_WINDOW_S, laps n-1..n+1 green and not neutralised, no pit-in/out of either car on n or n+1, no lapped
   cars. Success = A crosses the line ahead of D on lap n+1 (line-order flip), which is what "a position swap not
   explained by a pit stop" means; swaps caused by a third car's pit stop cannot happen because D is the car
   immediately ahead. Covariates: pace_delta (D's minus A's race pace, s/lap, positive = A faster; pace = driver-race
   fixed effect from a clean-air lap-time regression with race-lap FE, compound dummies and tyre age, so it is the
   tyre/fuel-normalised car pace the simulator takes from qualifying/practice -- but here it is estimated with
   hindsight on the same race, an optimistic input), tyre_age_delta (D.age - A.age, laps), gap (s at end of lap n),
   per-track offset (ridge). Fitted by penalised Newton (no sklearn in the venv).
   The per-track offset cannot be validated leave-one-race-out in a one-race-per-track season, so it is tested by
   transfer: offsets fitted on 2026 are scored on the same-track 2025 races.
3. Pass time cost. Extra regression columns on the dirty-air regression: attacker-on-pass-lap and defender-on-pass-lap
   indicators, i.e. lap-time cost of the pass lap relative to clean air AFTER the gap-bin terms (so it is on top of
   whatever dirty-air loss the lap already carries).
4. Lap 1: (end-of-lap-1 position - grid position) by grid bin from results.csv grid_position (pit-lane starters and
   missing grid dropped). Lap-1 SC/red races are kept (positions still change).
5. Minimum following gap: 5th percentile of the on-road gap among sustained followers (gap < 2 s at both the start and
   end of the lap). Timing-line resolution makes the true minimum smaller.

Anything not detectable stays a documented fallback; see "notes" in the JSON and team/handoffs/T-053-data-scientist.md.
"""

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, "backend")

import numpy as np
import pandas as pd

from app.core import tire_model as tm

PROCESSED = Path("data/processed")
STAGING = Path("data/staging")
OUT_PATH = Path("data/sim_params/race_interaction.json")
TRACK_PARAMS = Path("data/tracks/track_params.json")

OVERTAKE_WINDOW_S = 1.5
GAP_BINS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]  # dirty-air gap bins, reference = >= 3 s / leader
GRID_BINS = {"1-3": (1, 3), "4-6": (4, 6), "7-10": (7, 10), "11-15": (11, 15), "16-22": (16, 22)}
PACE_DELTA_CLIP_S = 2.0
PRIOR_2025_WEIGHT = 0.3
RIDGE_COEF = 0.05
OFFSET_LAMBDA = 0.5  # prior sd ~1 logit; the 2025 same-track transfer test prefers weak shrinkage (0.5 > 2 > 8 > inf)
RNG = np.random.default_rng(20261116)


# ---------------------------------------------------------------- loading
def load_races(season: int) -> pd.DataFrame:
    """All dry laps of a season with race_id, grid position and the columns used below."""
    frames = []
    if season == 2026:
        paths = {p.name: p for p in sorted(STAGING.glob("*_2026_laps.csv"))}
        paths.update({p.name: p for p in sorted(PROCESSED.glob("*_2026_laps.csv"))})  # processed wins
        files = sorted(paths.values(), key=lambda p: p.name)
    else:
        files = [Path(p) for p in glob.glob(str(STAGING / "*_2025_laps.csv"))]
        files = [p for p in sorted(files) if p.name.replace("_laps.csv", "") not in tm.EXCLUDED_2025_RACES]
    for p in files:
        df = pd.read_csv(p)
        df["race_file"] = str(p)
        df["race_id"] = p.name.replace("_laps.csv", "")
        res = p.with_name(p.name.replace("_laps.csv", "_results.csv"))
        if res.exists():
            g = pd.read_csv(res)[["driver", "grid_position"]]
            df = df.merge(g, on="driver", how="left")
        else:
            df["grid_position"] = np.nan
        frames.append(df)
    laps = pd.concat(frames, ignore_index=True)
    laps = tm.drop_wet_races(laps)
    laps["track_id"] = laps["race_file"].map(tm.track_id_of_race)
    return laps.sort_values(["race_id", "driver", "lap_number"]).reset_index(drop=True)


def add_gap_columns(laps: pd.DataFrame) -> pd.DataFrame:
    """ahead_gap: s to the car that crossed the line just before, same lap only. inf for leaders (the previous
    crossing is an earlier lap), NaN when a lapped car is directly ahead (blue-flag traffic, excluded)."""
    d = laps["road_ahead_lap_diff"]
    g = laps["road_gap_ahead_s"]
    laps["ahead_gap"] = np.where(d == 0, g, np.where(d < 0, np.inf, np.nan))
    laps.loc[g.isna(), "ahead_gap"] = np.nan
    laps.loc[laps["lap_number"] == 1, "ahead_gap"] = np.nan  # standing-start grid slots, not racing gaps
    laps["prev_gap"] = laps.groupby(["race_id", "driver"])["ahead_gap"].shift(1)
    return laps


# ---------------------------------------------------------------- fixed-effect machinery
def demean(frame: pd.DataFrame, cols: list, keys: list, iters: int = 60) -> pd.DataFrame:
    """Remove two nested-in-race fixed effects (alternating projections) from cols."""
    x = frame[cols].astype(float).copy()
    for _ in range(iters):
        before = x.values.copy()
        for k in keys:
            x = x - x.groupby([frame[c] for c in k]).transform("mean")
        if np.abs(x.values - before).max() < 1e-9:
            break
    return x


def ols(X: np.ndarray, y: np.ndarray):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def boot_ols(X: np.ndarray, y: np.ndarray, races: np.ndarray, n_boot: int):
    """Cluster (race) bootstrap of an OLS on already-demeaned data -> (beta, se, array of draws)."""
    beta = ols(X, y)
    ids = np.unique(races)
    idx = {r: np.where(races == r)[0] for r in ids}
    draws = []
    for _ in range(n_boot):
        pick = RNG.choice(ids, size=len(ids), replace=True)
        rows = np.concatenate([idx[r] for r in pick])
        draws.append(ols(X[rows], y[rows]))
    draws = np.array(draws)
    return beta, draws.std(axis=0), draws


# ---------------------------------------------------------------- pace per driver-race
def driver_race_pace(laps: pd.DataFrame) -> pd.Series:
    """Tyre/fuel-normalised pace per (race, driver), s/lap relative to the race mean (lower = faster)."""
    clean = tm.clean_laps(laps)
    clean = clean[clean["ahead_gap"].fillna(0) >= 3.0].copy()  # clean air only (NaN gap -> excluded below)
    clean = clean[clean["ahead_gap"].notna()]
    # damaged / off-track laps would drag a driver's fixed effect by tens of seconds (seen: -36 s): keep laps within 4%
    # of the race median of clean laps
    med = clean.groupby("race_id")["lap_time_seconds"].transform("median")
    clean = clean[clean["lap_time_seconds"] < 1.04 * med]
    out = {}
    for rid, r in clean.groupby("race_id"):
        drivers = sorted(r["driver"].unique())
        lap_ids = sorted(r["lap_number"].unique())
        d_ix = {d: i for i, d in enumerate(drivers)}
        l_ix = {l: i for i, l in enumerate(lap_ids)}
        n = len(r)
        cols = [("drv", len(drivers)), ("lap", len(lap_ids))]
        X = np.zeros((n, len(drivers) + len(lap_ids) + 3))
        X[np.arange(n), r["driver"].map(d_ix).values] = 1.0
        X[np.arange(n), len(drivers) + r["lap_number"].map(l_ix).values] = 1.0
        X[:, -3] = (r["compound"] == "SOFT").astype(float).values
        X[:, -2] = (r["compound"] == "MEDIUM").astype(float).values
        X[:, -1] = r["tire_age"].values
        beta = ols(X, r["lap_time_seconds"].values)
        fe = beta[: len(drivers)]
        fe = fe - fe.mean()
        cnt = r["driver"].value_counts()
        for d, v in zip(drivers, fe):
            if cnt.get(d, 0) >= 8:
                out[(rid, d)] = v
    return pd.Series(out, name="pace_s")


# ---------------------------------------------------------------- 1 + 3: dirty air and pass cost
def build_pass_events(laps: pd.DataFrame, window: float) -> pd.DataFrame:
    """One row per (attacker, lap n): see module docstring item 2. Also used for pass-lap indicators."""
    L = laps.copy()
    g = L.groupby(["race_id", "driver"])
    for c in ("is_neutralised",):
        L[c] = L[c].astype(bool)
    L["neut_prev"] = g["is_neutralised"].shift(1).astype("boolean").fillna(False).astype(bool)
    key = ["race_id", "driver", "lap_number"]
    keep = ["race_id", "driver", "lap_number", "lap_end_s", "tire_age", "pit_in_lap", "pit_out_lap", "is_green_flag",
            "is_neutralised", "neut_prev", "position", "compound"]
    base = L[keep]
    nxt = base.copy()
    nxt["lap_number"] = nxt["lap_number"] - 1  # row for lap n holds the n+1 values
    nxt = nxt.add_suffix("_n1").rename(columns={"race_id_n1": "race_id", "driver_n1": "driver", "lap_number_n1": "lap_number"})
    A = L[L["ahead_gap"].between(0, window) & (L["lap_number"] >= 2)].copy()
    A["defender"] = A["road_car_ahead"]
    A = A.merge(nxt, on=key, how="inner")
    D0 = base.add_prefix("d_").rename(columns={"d_race_id": "race_id", "d_driver": "defender", "d_lap_number": "lap_number"})
    A = A.merge(D0, on=["race_id", "defender", "lap_number"], how="inner")
    D1 = nxt.add_prefix("d_").rename(columns={"d_race_id": "race_id", "d_driver": "defender", "d_lap_number": "lap_number"})
    A = A.merge(D1, on=["race_id", "defender", "lap_number"], how="inner")
    ok = (
        A["is_green_flag"].astype(bool) & A["is_green_flag_n1"].astype(bool)
        & A["d_is_green_flag"].astype(bool) & A["d_is_green_flag_n1"].astype(bool)
        & ~A["is_neutralised"] & ~A["is_neutralised_n1"] & ~A["neut_prev"]
        & ~A["d_is_neutralised"] & ~A["d_is_neutralised_n1"] & ~A["d_neut_prev"]
        & (A[["pit_in_lap", "pit_out_lap", "pit_in_lap_n1", "pit_out_lap_n1", "d_pit_in_lap", "d_pit_out_lap",
              "d_pit_in_lap_n1", "d_pit_out_lap_n1"]].sum(axis=1) == 0)
        & A["lap_end_s"].notna() & A["lap_end_s_n1"].notna() & A["d_lap_end_s"].notna() & A["d_lap_end_s_n1"].notna()
        & (A["lap_end_s"] > A["d_lap_end_s"])  # A really is behind at the end of lap n
        & A["tire_age"].notna() & A["d_tire_age"].notna()
    )
    A = A[ok].copy()
    A["passed"] = (A["lap_end_s_n1"] < A["d_lap_end_s_n1"]).astype(int)
    A["tyre_age_delta"] = A["d_tire_age"] - A["tire_age"]
    A["gap"] = A["ahead_gap"]
    return A[["race_id", "track_id", "driver", "defender", "lap_number", "passed", "tyre_age_delta", "gap"]]


def fit_dirty_air(laps: pd.DataFrame, events: pd.DataFrame, n_boot: int) -> dict:
    clean = tm.clean_laps(laps)
    clean = clean[clean["ahead_gap"].notna()].copy()
    # prev_gap is NaN on lap 2 and after any pit/neutralised lap's missing gap; keep rows, it is only used for "sustained".
    gap = clean["ahead_gap"].values
    for lo, hi in zip(GAP_BINS[:-1], GAP_BINS[1:]):
        clean[f"bin_{lo}_{hi}"] = ((gap >= lo) & (gap < hi)).astype(float)
    clean["stint_key"] = clean["race_id"] + "|" + clean["driver"] + "|" + clean["tyre_stint"].astype(int).astype(str)
    clean["racelap_key"] = clean["race_id"] + "|" + clean["lap_number"].astype(int).astype(str)
    # pass-lap indicators (the lap on which the flip is recorded, n+1)
    wins = events[events["passed"] == 1]
    att = set(zip(wins["race_id"], wins["driver"], wins["lap_number"] + 1))
    dfd = set(zip(wins["race_id"], wins["defender"], wins["lap_number"] + 1))
    ident = list(zip(clean["race_id"], clean["driver"], clean["lap_number"]))
    clean["pass_att"] = [float(i in att) for i in ident]
    clean["pass_def"] = [float(i in dfd) for i in ident]

    bin_cols = [f"bin_{lo}_{hi}" for lo, hi in zip(GAP_BINS[:-1], GAP_BINS[1:])]
    keys = [["stint_key"], ["racelap_key"]]
    races = clean["race_id"].values

    out = {"n_laps": int(len(clean)), "n_races": int(clean["race_id"].nunique())}

    # (a) bin profile
    # tyre age is NOT a regressor here: age = lap - stint start, so stint FE + race-lap FE absorb it exactly
    # (an earlier version included it and got a meaningless -0.22 s/lap slope).
    names = bin_cols + ["pass_att", "pass_def"]
    dm = demean(clean, ["lap_time_seconds"] + names, keys)
    X = dm[names].values
    y = dm["lap_time_seconds"].values
    beta, se, _ = boot_ols(X, y, races, n_boot)
    out["profile"] = {n: (float(b), float(s)) for n, b, s in zip(names, beta, se)}
    out["n_in_bin"] = {c: int(clean[c].sum()) for c in bin_cols}
    return out, clean, keys


def fit_window_model(clean: pd.DataFrame, keys: list, window: float, n_boot: int, sustained: bool = False,
                     fuel_coef: float = -0.0512) -> dict:
    """Loss: two-way FE, lap time on one in-window dummy. Extra wear: one-way (stint) FE on fuel-corrected lap time
    (y - fuel_coef * lap) so tyre age is identified; extra_deg_frac = (dirty x age slope) / (age slope)."""
    c = clean.copy()
    gap = c["ahead_gap"].values
    dirty = (gap < window)
    if sustained:
        dirty &= (c["prev_gap"].values < window)
    c["dirty"] = dirty.astype(float)
    c["dirty_age"] = c["dirty"] * c["tire_age"]
    races = c["race_id"].values
    dm = demean(c, ["lap_time_seconds", "dirty"], keys)
    beta, se, _ = boot_ols(dm[["dirty"]].values, dm["lap_time_seconds"].values, races, n_boot)
    c["y_fuel"] = c["lap_time_seconds"] - fuel_coef * c["lap_number"]
    dm2 = demean(c, ["y_fuel", "tire_age", "dirty", "dirty_age"], [keys[0]])
    b2, se2, draws = boot_ols(dm2[["tire_age", "dirty", "dirty_age"]].values, dm2["y_fuel"].values, races, n_boot)
    return {"window": window, "sustained": sustained, "n_dirty": int(c["dirty"].sum()),
            "dirty_loss_s": float(beta[0]), "dirty_loss_se": float(se[0]),
            "deg_s_per_lap": float(b2[0]), "dirty_x_age": float(b2[2]), "dirty_x_age_se": float(se2[2]),
            "extra_deg_frac": float(b2[2] / b2[0]) if b2[0] > 1e-4 else float("nan"),
            "extra_deg_frac_draws": draws[:, 2] / np.where(draws[:, 0] > 1e-3, draws[:, 0], np.nan)}


# ---------------------------------------------------------------- 2: overtake logistic
def fit_logistic(X: np.ndarray, y: np.ndarray, w: np.ndarray, track_ix: np.ndarray, n_tracks: int,
                 lam_off: float = OFFSET_LAMBDA, ridge: float = RIDGE_COEF):
    """Penalised IRLS. Params = [intercept, pace, age, gap, offset_1..K]; offsets and the three slopes ridge-shrunk."""
    n, p = X.shape
    k = p + 1 + n_tracks  # intercept + p slopes + offsets
    Z = np.zeros((n, k))
    Z[:, 0] = 1.0
    Z[:, 1 : 1 + p] = X
    if n_tracks:
        Z[np.arange(n), 1 + p + track_ix] = 1.0
    pen = np.zeros(k)
    pen[1 : 1 + p] = ridge
    pen[1 + p :] = lam_off
    beta = np.zeros(k)
    for _ in range(50):
        eta = np.clip(Z @ beta, -30, 30)
        mu = 1 / (1 + np.exp(-eta))
        W = w * mu * (1 - mu)
        grad = Z.T @ (w * (y - mu)) - pen * beta
        H = (Z * W[:, None]).T @ Z + np.diag(pen) + 1e-9 * np.eye(k)
        step = np.linalg.solve(H, grad)
        beta = beta + step
        if np.abs(step).max() < 1e-8:
            break
    return beta


def predict_logit(beta, X, track_ix=None, n_tracks=0):
    p = X.shape[1]
    eta = beta[0] + X @ beta[1 : 1 + p]
    if n_tracks and track_ix is not None:
        eta = eta + np.where(track_ix >= 0, beta[1 + p :][np.clip(track_ix, 0, None)], 0.0)
    return 1 / (1 + np.exp(-np.clip(eta, -30, 30)))


def logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


FEATS = ["pace_delta", "tyre_age_delta", "gap"]


def design(ev: pd.DataFrame, feats=FEATS) -> np.ndarray:
    return ev[feats].values.astype(float)


def loro(ev26: pd.DataFrame, ev25: pd.DataFrame, feats=FEATS) -> dict:
    """Leave-one-race-out over 2026 races; held-out race scored with offset 0 (a track unseen in training)."""
    rows = []
    for race in sorted(ev26["race_id"].unique()):
        tr = ev26[ev26["race_id"] != race]
        te = ev26[ev26["race_id"] == race]
        if te["passed"].sum() == 0 or len(te) < 30:
            continue
        tracks = sorted(tr["track_id"].unique())
        t_ix = tr["track_id"].map({t: i for i, t in enumerate(tracks)}).values
        out = {"race": race, "n": len(te), "passes": int(te["passed"].sum())}
        y = te["passed"].values.astype(float)
        b = fit_logistic(design(tr, feats), tr["passed"].values.astype(float), np.ones(len(tr)), t_ix, len(tracks))
        out["ll_model"] = logloss(y, predict_logit(b, design(te, feats)))
        out["ll_const"] = logloss(y, np.full(len(te), tr["passed"].mean()))
        # model without track offsets
        b0 = fit_logistic(design(tr, feats), tr["passed"].values.astype(float), np.ones(len(tr)), t_ix * 0, 0)
        out["ll_nooff"] = logloss(y, predict_logit(b0, design(te, feats)))
        if len(ev25):
            pool = pd.concat([tr, ev25], ignore_index=True)
            w = np.concatenate([np.ones(len(tr)), np.full(len(ev25), PRIOR_2025_WEIGHT)])
            tr_all = sorted(pool["track_id"].unique())
            ix = pool["track_id"].map({t: i for i, t in enumerate(tr_all)}).values
            b2 = fit_logistic(design(pool, feats), pool["passed"].values.astype(float), w, ix, len(tr_all))
            out["ll_with2025"] = logloss(y, predict_logit(b2, design(te, feats)))
        rows.append(out)
    return pd.DataFrame(rows)


def summarise_loro(df: pd.DataFrame, a: str, b: str) -> str:
    """mean(b - a) log-loss over races with a race-level standard error; negative = a is better."""
    d = (df[a] - df[b]).values
    return f"{d.mean():+.4f} +/- {d.std(ddof=1) / np.sqrt(len(d)):.4f} (n={len(d)} races)"


# ---------------------------------------------------------------- 4: lap 1
def fit_lap1(laps: pd.DataFrame) -> dict:
    l1 = laps[(laps["lap_number"] == 1) & laps["position"].notna() & laps["grid_position"].notna()].copy()
    l1 = l1[l1["grid_position"] >= 1]
    l1["delta"] = l1["position"] - l1["grid_position"]
    res = {}
    for name, (lo, hi) in GRID_BINS.items():
        s = l1[(l1["grid_position"] >= lo) & (l1["grid_position"] <= hi)]["delta"]
        res[name] = {"mean": round(float(s.mean()), 3), "sd": round(float(s.std(ddof=1)), 3), "n": int(len(s))}
    return res, int(l1["race_id"].nunique())


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    n_boot = 100 if args.quick else 400

    laps26 = add_gap_columns(load_races(2026))
    laps25 = add_gap_columns(load_races(2025))
    races26 = sorted(laps26["race_id"].unique())
    print(f"2026 dry races: {len(races26)} {races26}")
    print(f"2025 dry races: {laps25['race_id'].nunique()}")

    # ---- 1/3 dirty air + pass cost
    ev_all26 = build_pass_events(laps26, OVERTAKE_WINDOW_S)
    prof, clean26, keys = fit_dirty_air(laps26, ev_all26, n_boot)
    print(f"\n[1] dirty-air sample: {prof['n_laps']} clean laps, {prof['n_races']} races")
    print("gap-bin loss profile (s vs gap >= 3 s; se from race bootstrap):")
    for k, (b, s) in prof["profile"].items():
        extra = f"  n={prof['n_in_bin'][k]}" if k in prof["n_in_bin"] else ""
        print(f"  {k:<14} {b:+.3f} +/- {s:.3f}{extra}")
    wm = {}
    for w in (0.5, 0.8, 1.0, 1.2, 1.5, 2.0):
        for sus in (False, True):
            r = fit_window_model(clean26, keys, w, n_boot=n_boot // 4, sustained=sus)
            wm[(w, sus)] = r
            print(f"  window {w:.1f}s sustained={sus!s:<5} n={r['n_dirty']:5d} loss {r['dirty_loss_s']:+.3f}+/-{r['dirty_loss_se']:.3f}"
                  f"  deg {r['deg_s_per_lap']:+.4f}  dirty*age {r['dirty_x_age']:+.4f}+/-{r['dirty_x_age_se']:.4f}"
                  f"  extra_deg_frac {r['extra_deg_frac']:+.2f}")
    W_CHOSEN = 1.0
    main_fit = fit_window_model(clean26, keys, W_CHOSEN, n_boot=n_boot, sustained=False)
    fr = main_fit["extra_deg_frac_draws"]
    fr = fr[np.isfinite(fr)]
    frac_lo, frac_hi = np.percentile(fr, [2.5, 97.5])
    print(f"CHOSEN window {W_CHOSEN}s: loss {main_fit['dirty_loss_s']:+.3f} +/- {main_fit['dirty_loss_se']:.3f} s, "
          f"extra_deg_frac {main_fit['extra_deg_frac']:+.2f} (95% boot {frac_lo:+.2f}..{frac_hi:+.2f})")

    # ---- 2 overtake
    pace = driver_race_pace(laps26)
    pace25 = driver_race_pace(laps25)

    def finish(ev, pace_s):
        ev = ev.copy()
        pa = [pace_s.get((r, d), np.nan) for r, d in zip(ev["race_id"], ev["driver"])]
        pd_ = [pace_s.get((r, d), np.nan) for r, d in zip(ev["race_id"], ev["defender"])]
        # positive = attacker faster; clipped at +-2 s/lap: bigger values are damaged cars, not pace
        ev["pace_delta"] = np.clip(np.array(pd_) - np.array(pa), -PACE_DELTA_CLIP_S, PACE_DELTA_CLIP_S)
        return ev.dropna(subset=["pace_delta"])

    laps26["track_id"] = laps26["race_file"].map(tm.track_id_of_race)
    laps25["track_id"] = laps25["race_file"].map(tm.track_id_of_race)
    tid26 = laps26.drop_duplicates("race_id").set_index("race_id")["track_id"]
    tid25 = laps25.drop_duplicates("race_id").set_index("race_id")["track_id"]
    ev26_raw = build_pass_events(laps26, OVERTAKE_WINDOW_S)
    ev26_raw["track_id"] = ev26_raw["race_id"].map(tid26)
    ev26 = finish(ev26_raw, pace)
    ev25_raw = build_pass_events(laps25, OVERTAKE_WINDOW_S)
    ev25_raw["track_id"] = ev25_raw["race_id"].map(tid25)
    ev25 = finish(ev25_raw, pace25)
    print(f"\n[2] overtake obs 2026: {len(ev26)} ({int(ev26['passed'].sum())} passes, rate {ev26['passed'].mean():.3f}); "
          f"2025: {len(ev25)} ({int(ev25['passed'].sum())} passes)")
    print(ev26.groupby("race_id")["passed"].agg(["size", "sum"]).T.to_string())

    lo = loro(ev26, ev25)
    print("\nLOO-race log-loss (held-out track offset = 0):")
    print(lo.round(4).to_string(index=False))
    print(f"  mean model {lo['ll_model'].mean():.4f}  no-offset {lo['ll_nooff'].mean():.4f}  const {lo['ll_const'].mean():.4f}"
          f"  with2025 {lo['ll_with2025'].mean():.4f}")
    print("  model - const :", summarise_loro(lo, "ll_model", "ll_const"))
    print("  nooff - const :", summarise_loro(lo, "ll_nooff", "ll_const"))
    print("  with2025 - model :", summarise_loro(lo, "ll_with2025", "ll_model"))
    use_2025 = bool((lo["ll_with2025"] - lo["ll_model"]).mean() < -2 * (lo["ll_with2025"] - lo["ll_model"]).std(ddof=1) / np.sqrt(len(lo)))
    print(f"  use 2025 prior: {use_2025}")

    # feature ablations (LOO), to state which coefficients carry the signal
    for name, feats in (("pace only", ["pace_delta"]), ("gap only", ["gap"]), ("age only", ["tyre_age_delta"]),
                        ("pace+gap", ["pace_delta", "gap"])):
        r = loro(ev26, pd.DataFrame(columns=ev26.columns), feats)
        print(f"  ablation {name:<10}: model - const {summarise_loro(r, 'll_nooff', 'll_const')}")

    # final fit on all 2026 (offsets from 2026 only unless the 2025 prior wins)
    pool = ev26
    w = np.ones(len(pool))
    if use_2025:
        pool = pd.concat([ev26, ev25], ignore_index=True)
        w = np.concatenate([np.ones(len(ev26)), np.full(len(ev25), PRIOR_2025_WEIGHT)])
    tracks = sorted(pool["track_id"].unique())
    t_ix = pool["track_id"].map({t: i for i, t in enumerate(tracks)}).values
    beta = fit_logistic(design(pool), pool["passed"].values.astype(float), w, t_ix, len(tracks))
    coefs = dict(zip(["intercept"] + FEATS, beta[:4]))
    offsets = {t: round(float(o), 4) for t, o in zip(tracks, beta[4:])}
    print("\nfinal logistic:", {k: round(float(v), 4) for k, v in coefs.items()})
    print("track offsets:", offsets)

    # coefficient CIs by race bootstrap
    ids = sorted(ev26["race_id"].unique())
    bs = []
    for _ in range(n_boot // 2):
        pick = RNG.choice(ids, size=len(ids), replace=True)
        b = pd.concat([ev26[ev26["race_id"] == r] for r in pick], ignore_index=True)
        bs.append(fit_logistic(design(b), b["passed"].values.astype(float), np.ones(len(b)), b["track_id"].map({t: i for i, t in enumerate(tracks)}).values, len(tracks))[:4])
    bs = np.array(bs)
    ci = {k: (float(np.percentile(bs[:, i], 2.5)), float(np.percentile(bs[:, i], 97.5))) for i, k in enumerate(["intercept"] + FEATS)}
    print("95% race-bootstrap CI:", {k: (round(a, 3), round(b, 3)) for k, (a, b) in ci.items()})

    # track-offset transfer test: 2026-fitted offsets scored on same-track 2025 races, vs lambda and vs no offsets
    print("\ntrack-offset transfer (fit 2026, score same-track 2025; log-loss, lower better):")
    shared = ev25[ev25["track_id"].isin(tracks)]
    y25 = shared["passed"].values.astype(float)
    t25 = shared["track_id"].map({t: i for i, t in enumerate(tracks)}).values
    transfer = {}
    for lam in (0.5, 2.0, 8.0, 1e6):
        b = fit_logistic(design(ev26), ev26["passed"].values.astype(float), np.ones(len(ev26)),
                         ev26["track_id"].map({t: i for i, t in enumerate(tracks)}).values, len(tracks), lam_off=lam)
        transfer[lam] = logloss(y25, predict_logit(b, design(shared), t25, len(tracks)))
        print(f"  lambda {lam:>9}: {transfer[lam]:.4f}  (n={len(shared)})")
    print(f"  constant rate: {logloss(y25, np.full(len(y25), ev26['passed'].mean())):.4f}")

    # ---- 3 pass cost
    pa = prof["profile"]["pass_att"]
    pdf = prof["profile"]["pass_def"]
    print(f"\n[3] pass-lap cost vs clean air, after gap-bin terms: attacker {pa[0]:+.3f}+/-{pa[1]:.3f}, defender {pdf[0]:+.3f}+/-{pdf[1]:.3f}"
          f" (n passes {int(ev_all26['passed'].sum())})")

    # ---- 4 lap 1
    lap1, n_l1_races = fit_lap1(laps26)
    print(f"\n[4] lap-1 position change by grid bin ({n_l1_races} races):", lap1)

    # ---- 5 min follow gap
    g = laps26[laps26["ahead_gap"].between(0, 2.0) & laps26["prev_gap"].between(0, 2.0)
               & laps26["is_green_flag"].astype(bool) & (laps26["pit_in_lap"] == 0) & (laps26["pit_out_lap"] == 0)]["ahead_gap"]
    pcts = {q: float(np.percentile(g, q)) for q in (1, 2, 5, 10, 25, 50)}
    print(f"\n[5] sustained-follower gap percentiles (n={len(g)}):", {k: round(v, 3) for k, v in pcts.items()})

    # ---- assemble
    loss = main_fit["dirty_loss_s"]
    extra = main_fit["extra_deg_frac"]
    extra_detectable = bool(np.isfinite(extra) and frac_lo > 0)
    out = {
        "version": 1,
        "fitted_on": races26,
        "dirty_air": {
            "window_s": W_CHOSEN,
            "time_loss_s": round(float(max(loss, 0.0)), 3),
            "extra_deg_frac": round(float(max(extra, 0.0)), 3) if extra_detectable else 0.0,
            "time_loss_se_s": round(main_fit["dirty_loss_se"], 3),
            "extra_deg_frac_raw": round(float(extra), 3),
            "extra_deg_frac_ci95": [round(float(frac_lo), 3), round(float(frac_hi), 3)],
            "extra_deg_detectable": extra_detectable,
            "loss_by_gap_bin_s": {k: round(v[0], 3) for k, v in prof["profile"].items() if k.startswith("bin_")},
        },
        "overtake": {
            "model": "logistic",
            "window_s": OVERTAKE_WINDOW_S,
            "intercept": round(float(coefs["intercept"]), 4),
            "pace_delta_coef": round(float(coefs["pace_delta"]), 4),
            "tyre_age_delta_coef": round(float(coefs["tyre_age_delta"]), 4),
            "gap_coef": round(float(coefs["gap"]), 4),
            "track_offset": offsets,
            "coef_ci95": {k: [round(a, 4), round(b, 4)] for k, (a, b) in ci.items()},
            "base_rate_per_lap": round(float(ev26["passed"].mean()), 4),
            "used_2025_prior": use_2025,
        },
        "pass_time_cost_s": {
            "attacker": round(float(pa[0]), 3),
            "defender": round(float(pdf[0]), 3),
            "attacker_se": round(float(pa[1]), 3),
            "defender_se": round(float(pdf[1]), 3),
        },
        "lap1": {"by_grid_bin": {k: {"mean": v["mean"], "sd": v["sd"]} for k, v in lap1.items()},
                 "n_by_grid_bin": {k: v["n"] for k, v in lap1.items()}},
        "min_follow_gap_s": round(pcts[5], 3),
        "n_obs": {
            "dirty_air_laps": prof["n_laps"],
            "dirty_air_laps_in_window": main_fit["n_dirty"],
            "overtake_obs": int(len(ev26)),
            "overtake_passes": int(ev26["passed"].sum()),
            "lap1_drivers": int(sum(v["n"] for v in lap1.values())),
            "follow_gap_laps": int(len(g)),
            "races": len(races26),
        },
        "validation": {
            "loro_logloss_model": round(float(lo["ll_model"].mean()), 4),
            "loro_logloss_const": round(float(lo["ll_const"].mean()), 4),
            "loro_logloss_model_no_offsets": round(float(lo["ll_nooff"].mean()), 4),
            "loro_logloss_with_2025_prior": round(float(lo["ll_with2025"].mean()), 4),
            "loro_model_minus_const": summarise_loro(lo, "ll_model", "ll_const"),
        },
        "notes": (
            "Fitted 2026-11-16 by fit_race_interaction.py from end-of-lap on-road gaps (one sample per lap). "
            "dirty_air.time_loss_s is seconds per lap for a car whose gap to the car ahead at the end of lap is below "
            "window_s, vs clean air, from two-way fixed effects (stint, race-lap). overtake probability is per lap for a "
            "car directly behind within overtake.window_s: p = sigmoid(intercept + pace_delta_coef*pace_delta + "
            "tyre_age_delta_coef*tyre_age_delta + gap_coef*gap + track_offset[track]); pace_delta = defender pace - "
            "attacker pace in s/lap (positive = attacker faster, tyre/fuel-normalised car pace), tyre_age_delta = defender "
            "age - attacker age (positive = attacker fresher). Use track_offset = 0 for tracks not listed. "
            "pass_time_cost_s is the extra lap time on the lap the pass is recorded, beyond the dirty-air term, "
            "relative to clean air (may be negative). lap1 mean/sd are (end-of-lap-1 position - grid position); positive "
            "= lost places. min_follow_gap_s is the 5th percentile of sustained-follower gaps at timing-line resolution."
        ),
    }
    if not args.no_write:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
