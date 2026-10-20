"""
Validation scorecard v2 (T-003): the numbers every later phase has to beat.

Run from the repo root:  python validate_scorecard.py
  --out PATH     where to write the JSON (default data/scorecard/latest.json)
  --no-write     print only

Scores the engine (tire_model + race_simulator, no noise) against the 10
cached 2026 races in data/processed/. Every race is scored with a tyre
model fitted on the OTHER nine (leave-one-race-out, LORO), and with a
base_laptime that doesn't use that race's own laps.

Sections (all computed per held-out race, then pooled):

1. base_laptime. race_simulator treats base_laptime as "HARD tyre, age 2,
   empty tank" for one driver at one track. validate_replay.py used the
   driver's own fastest lap of the race being scored, which leaks the
   answer. Three methods are scored side by side:
     - quali_ratio (PRIMARY, non-leaky): driver's best qualifying time
       (min of Q1/Q2/Q3, from the per-round Jolpica cache files written by
       jolpica_client in data/jolpica/raw/ -- read-only, no network) x a race/quali ratio. The ratio is the
       pooled median, over the 9 TRAINING races, of
       oracle_base / quali_best, where oracle_base is the base_laptime
       that makes the (LORO) model match that driver's clean green laps
       (median residual). So the ratio is calibrated against the
       simulator's own definition of base_laptime, not against "fastest
       lap", and never touches the held-out race's laps.
     - fastest_lap (LEAKY, the old validate_replay method): reference only.
     - oracle (LEAKY, diagnostic floor): base fitted on the held-out
       race's own clean laps. What's left is error from the tyre/fuel
       shape, SC laps, etc. -- the error the quali method would have if
       it predicted the level perfectly.
   Why qualifying and not something else (2026-10-19): the FastF1 cache
   holds no 2026 qualifying sessions (the Qualifying dirs under
   cache/2026/ are empty -- the T-009 warm-up hit the 401) and
   results.csv's q1_s/q2_s/q3_s columns are empty, but Jolpica has
   classified Q1/Q2/Q3 times for every round. Qualifying is the only
   per-driver pace signal from the same weekend that is known before the
   race. Prior-race race pace can't set the per-track level (every track
   is different), so it could only refine the per-driver offset -- left
   for later. Measured race/quali ratio of the fastest lap ranges
   1.023 (Canada) .. 1.067 (Hungary/Austria) by track, so one pooled
   ratio costs about +-1.5% (~1 s/lap) in level at the extremes; that is
   the main price of not leaking, and it shows up as per-stint bias.
   Drivers with no qualifying time (Australia VER/STR/SAI, one at Miami)
   get their teammate's time, else the session's slowest; flagged in
   the JSON.

2. Per-stint mean s/lap error. Each driver's real strategy
   (validate_replay.reconstruct_strategy: real tyre stints + start ages)
   is simulated over the full race distance -- the last stint is extended
   to the race's lap count so fuel_effect is anchored the same way for
   lapped cars and DNFs -- and compared lap-by-lap with the laps the
   driver actually ran. Per stint, error = mean(sim - actual) over the
   evaluated laps (>= MIN_EVAL_LAPS of them). Lap 1 (standing start) and
   pit in/out laps are excluded in every variant: the engine charges the
   whole pit loss to the out-lap while reality splits it across in- and
   out-lap, which would bias per-stint means by +-10 s/lap on those laps
   without saying anything about tyres or fuel. Variants:
     - incl_neutralised: every other lap, SC/VSC/red included (laps with
       no LapTime use lap_end_s - lap_start_s; red-flag laps can be
       ~2000 s, so read the median too).
     - excl_neutralised: also drop is_neutralised (SC/VSC/red) laps.
       This is the headline: the engine has no safety cars yet.
     - clean: tire_model.clean_laps rules (also drops restart laps and
       local-yellow laps) -- the population the tyre model is fitted on.
   Reported: bias (mean signed), MAE, median |err|, RMSE, plus
   "relative" MAE: each stint's error minus that driver's mean stint
   error (drivers with >= 2 stints). Level errors cancel when comparing
   two strategies for the same car, so relative MAE is the part that
   matters for ranking strategies.
   Drivers whose strategy can't be simulated (unknown tyres -- Hungary
   PER; INTERMEDIATE stints -- Canada) are skipped and counted.

3. Finishing-order rank correlation. The engine is one-car-only, so this
   simulates each classified finisher's (Finished/Lapped) real strategy
   independently over the full distance with their own base_laptime and
   ranks the total times -- no traffic, no SC, no overtaking. Spearman
   rho vs results.csv position, per race, against the grid-order
   baseline (results.csv grid_position) on exactly the same drivers.

4. Strategy-ranking accuracy. Pairs of classified finishers in the same
   race with SIMILAR PACE (|quali_best difference| <= SIMILAR_PACE_PCT
   of the faster one's time -- pre-race information only) on DIFFERENT
   strategies (different compound sequence / number of stops). The sim
   "predicts" the pair right if the driver with the lower simulated total
   finished ahead. Reported: accuracy with a Wilson 95% CI and a
   race-cluster bootstrap 95% CI, against the grid-order baseline and a
   strategy-only sim (same base_laptime for both, i.e. only the strategy
   differs) on the same pairs. A "broad" set also admits same-sequence
   pairs whose stop laps differ by >= TIMING_DIFF_LAPS. Pairs share
   drivers, so they are not independent -- the Wilson CI is optimistic;
   the race bootstrap is the honest one. The sample is small.

5. Race totals (context, comparable to validate_replay.py): per Finished
   driver, sim total vs real timing-line total, all laps and with
   SC/VSC/red laps removed from both.

Every number here is deterministic (no Monte Carlo noise; the bootstrap
has a fixed seed), so two runs on the same data give the same JSON.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.core import jolpica_client, tire_model as tm  # noqa: E402
from app.core.race_simulator import DEFAULT_PIT_LOSS_SECONDS, Stint, simulate_race  # noqa: E402
from validate_replay import actual_race_time, reconstruct_strategy  # noqa: E402

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
JOLPICA_RAW_DIR = REPO_ROOT / "data" / "jolpica" / "raw"
DEFAULT_OUT = REPO_ROOT / "data" / "scorecard" / "latest.json"
SCORECARD_VERSION = 2

YEAR = 2026
MAX_ROUND = 30  # upper bound on rounds probed in the Jolpica per-round cache
BASE_METHODS = ("quali_ratio", "fastest_lap", "oracle")
PRIMARY_METHOD = "quali_ratio"
LEAKY_METHODS = ("fastest_lap", "oracle")
VARIANTS = ("incl_neutralised", "excl_neutralised", "clean")
HEADLINE_VARIANT = "excl_neutralised"
MIN_EVAL_LAPS = 3
CLASSIFIED_STATUSES = ("Finished", "Lapped")
# "Similar pace" for strategy-ranking pairs: qualifying best times within
# 0.5% (~0.4 s on an 80 s lap). Field spread in 2026 qualifying is ~3-4%,
# so this keeps neighbours on pace while leaving enough pairs to count.
SIMILAR_PACE_PCT = 0.005
TIMING_DIFF_LAPS = 5
BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 20261019


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------


@dataclass
class RaceData:
    gp: str
    laps: pd.DataFrame  # one race's laps CSV (+ race_file column)
    results: pd.DataFrame
    info: dict
    quali: pd.DataFrame = field(default_factory=pd.DataFrame)  # driver, q_best, q_source

    @property
    def total_laps(self) -> int:
        return int(self.info["total_laps"])


def race_names(processed_dir: Path = PROCESSED_DIR) -> list[str]:
    return sorted(
        Path(p).name.split(f"_{YEAR}_")[0]
        for p in glob.glob(str(processed_dir / f"*_{YEAR}_laps.csv"))
    )


def _lap_time_str_to_s(value: Optional[str]) -> float:
    if not value:
        return np.nan
    minutes, seconds = value.split(":")
    return int(minutes) * 60 + float(seconds)


def _read_cached_pages(path: str, raw_dir: Path) -> list[dict]:
    """
    Cached Jolpica pages for one endpoint, named exactly as jolpica_client
    names them (jolpica_client._cache_path), but read from `raw_dir`.
    Read-only by construction: no HTTP, no mkdir, nothing written (the
    client's own get_pages(offline=True) would mkdir RAW_DIR and is bound
    to its module-level directory, which the tests can't redirect).
    """
    pages, offset = [], 0
    while True:
        cp = raw_dir / jolpica_client._cache_path(path, offset).name
        if not cp.exists():
            break
        payload = json.loads(cp.read_text(encoding="utf-8"))
        pages.append(payload)
        offset += jolpica_client.PAGE_LIMIT
        if offset >= int(payload["MRData"].get("total", 0)):
            break
    return pages


def load_jolpica_qualifying(event_name: str, raw_dir: Path = JOLPICA_RAW_DIR) -> pd.DataFrame:
    """
    Classified qualifying (driver code, Q1-Q3, best) for one 2026 event, read
    from the Jolpica cache written by jolpica_client (T-011). The client
    caches the CURRENT season per round ("2026/<round>/qualifying" ->
    2026_<round>_qualifying__o0.json), so every round's file is tried, and
    the season-wide file ("2026/qualifying") last for completeness. Matched
    on raceName == race_info event_name. Never triggers a request. Returns
    an empty frame if the event isn't cached.
    (Fix 2026-10-19, T-012 QA: this used to glob a path the client never
    writes for 2026, so it silently found nothing.)
    """
    paths = [f"{YEAR}/{rnd}/qualifying" for rnd in range(1, MAX_ROUND + 1)] + [f"{YEAR}/qualifying"]
    for path in paths:
        for payload in _read_cached_pages(path, raw_dir):
            for race in payload["MRData"].get("RaceTable", {}).get("Races", []):
                if race.get("raceName") != event_name:
                    continue
                rows = []
                for r in race.get("QualifyingResults", []):
                    q = [_lap_time_str_to_s(r.get(k)) for k in ("Q1", "Q2", "Q3")]
                    rows.append(
                        {
                            "driver": r["Driver"].get("code"),
                            "quali_position": int(r["position"]),
                            "q1_s": q[0],
                            "q2_s": q[1],
                            "q3_s": q[2],
                            "q_best": np.nanmin(q) if not all(np.isnan(q)) else np.nan,
                        }
                    )
                return pd.DataFrame(rows)
    return pd.DataFrame(columns=["driver", "quali_position", "q1_s", "q2_s", "q3_s", "q_best"])


def quali_pace(quali: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """
    Per-driver qualifying best for every driver in results, with a fallback
    for drivers without a time: teammate's best, else the session's slowest.
    Columns: driver, q_best, q_source ("own" | "teammate" | "slowest" | "none").
    """
    own = quali.dropna(subset=["q_best"]).set_index("driver")["q_best"] if len(quali) else pd.Series(dtype=float)
    slowest = float(own.max()) if len(own) else np.nan
    rows = []
    for _, r in results.iterrows():
        drv = r["driver"]
        if drv in own.index:
            rows.append((drv, float(own[drv]), "own"))
            continue
        mates = results[(results["team"] == r["team"]) & (results["driver"] != drv)]["driver"]
        mate_times = [own[m] for m in mates if m in own.index]
        if mate_times:
            rows.append((drv, float(min(mate_times)), "teammate"))
        elif not np.isnan(slowest):
            rows.append((drv, slowest, "slowest"))
        else:
            rows.append((drv, np.nan, "none"))
    return pd.DataFrame(rows, columns=["driver", "q_best", "q_source"])


def load_race(gp: str, processed_dir: Path = PROCESSED_DIR, raw_dir: Path = JOLPICA_RAW_DIR) -> RaceData:
    laps_path = processed_dir / f"{gp}_{YEAR}_laps.csv"
    laps = pd.read_csv(laps_path)
    laps["race_file"] = str(laps_path)
    results = pd.read_csv(processed_dir / f"{gp}_{YEAR}_results.csv")
    with open(processed_dir / f"{gp}_{YEAR}_race_info.json", encoding="utf-8") as fh:
        info = json.load(fh)
    raw_quali = load_jolpica_qualifying(info["event_name"], raw_dir)
    return RaceData(gp=gp, laps=laps, results=results, info=info, quali=quali_pace(raw_quali, results))


# --------------------------------------------------------------------------
# per-driver simulation pieces
# --------------------------------------------------------------------------


def full_distance_strategy(driver_laps: pd.DataFrame, total_laps: int) -> list[Stint]:
    """
    The driver's real strategy, with the last stint extended to the race
    distance (lapped cars / DNFs), so the simulated fuel load on lap k
    matches a car that started with fuel for the whole race. Raises
    ValueError if a stint's tyre is unknown.
    """
    strategy = reconstruct_strategy(driver_laps)
    run = sum(s.laps for s in strategy)
    if run < total_laps:
        last = strategy[-1]
        strategy[-1] = Stint(last.compound, last.laps + (total_laps - run), last.start_age)
    return strategy


def actual_lap_times(driver_laps: pd.DataFrame) -> np.ndarray:
    """LapTime, or lap_end_s - lap_start_s where FastF1 has no LapTime (SC/red laps)."""
    d = driver_laps.sort_values("lap_number")
    t = d["lap_time_seconds"].to_numpy(dtype=float)
    fallback = (d["lap_end_s"] - d["lap_start_s"]).to_numpy(dtype=float)
    return np.where(np.isnan(t), fallback, t)


@dataclass
class DriverSim:
    """One driver in one race under one tyre model. Sim laps = component + base."""

    driver: str
    laps: pd.DataFrame  # driver's laps, sorted, index reset
    strategy: list  # full-distance Stint list
    component: np.ndarray  # model lap time with base_laptime = 0, full distance
    actual: np.ndarray  # actual lap times for the laps run
    clean_mask: np.ndarray  # tire_model.clean_laps membership per lap run
    stint_index: np.ndarray  # 0-based stint number per lap run

    def sim_total(self, base: float) -> float:
        return float(self.component.sum() + base * len(self.component))


def simulatable(strategy: list, model: dict) -> bool:
    return all(s.compound in model and s.compound in model.get("_compound_offsets", {}) for s in strategy)


def build_driver_sims(race: RaceData, model: dict) -> tuple[dict, dict]:
    """
    DriverSim for every driver whose real strategy the engine can run.
    Returns (sims by driver, skipped {driver: reason}).
    """
    laps = race.laps
    clean_idx = set(tm.clean_laps(tm.assign_stint_ids(laps)).index)
    sims, skipped = {}, {}
    for drv, d in laps.groupby("driver"):
        d = d.sort_values("lap_number")
        try:
            strategy = full_distance_strategy(d, race.total_laps)
        except ValueError as exc:
            skipped[drv] = f"unknown tyre: {exc}"
            continue
        if not simulatable(strategy, model):
            skipped[drv] = "compound not modelled: " + ",".join(sorted({s.compound for s in strategy} - set(tm.DRY_COMPOUNDS)))
            continue
        comp = np.asarray(simulate_race(strategy, base_laptime=0.0, tire_model=model).lap_times)
        # stint number for each lap actually run, from the same stint key reconstruct_strategy uses
        key = d["tyre_stint"] if "tyre_stint" in d.columns else (d["tire_age"] == 1).cumsum()
        stint_index = pd.factorize(key, sort=True)[0]
        sims[drv] = DriverSim(
            driver=drv,
            laps=d.reset_index(drop=False),
            strategy=strategy,
            component=comp,
            actual=actual_lap_times(d),
            clean_mask=np.array([i in clean_idx for i in d.index]),
            stint_index=stint_index,
        )
    return sims, skipped


def oracle_base(sim: DriverSim) -> float:
    """base_laptime that zeroes the median residual on the driver's clean laps (NaN if none)."""
    n = len(sim.actual)
    m = sim.clean_mask
    if not m.any():
        return np.nan
    return float(np.median(sim.actual[m] - sim.component[:n][m]))


# --------------------------------------------------------------------------
# base_laptime
# --------------------------------------------------------------------------


def fit_quali_ratio(train: list[tuple[RaceData, dict]]) -> dict:
    """
    Pooled median of oracle_base / q_best over training races (each race's
    sims built with the model passed alongside it -- the held-out race's
    LORO model; training races are its training data, so nothing leaks).
    Only drivers with their own qualifying time count.
    """
    ratios, per_race = [], {}
    for race, sims in train:
        q = race.quali.set_index("driver")
        rr = []
        for drv, s in sims.items():
            if drv not in q.index or q.at[drv, "q_source"] != "own":
                continue
            ob = oracle_base(s)
            if np.isnan(ob):
                continue
            rr.append(ob / q.at[drv, "q_best"])
        ratios += rr
        per_race[race.gp] = float(np.median(rr)) if rr else None
    return {"ratio": float(np.median(ratios)), "n": len(ratios), "per_training_race": per_race}


def base_laptimes(race: RaceData, sims: dict, quali_ratio: Optional[float]) -> dict:
    """{method: {driver: base_laptime}} for every simulatable driver."""
    q = race.quali.set_index("driver")["q_best"]
    out = {m: {} for m in BASE_METHODS}
    for drv, s in sims.items():
        if quali_ratio is not None and drv in q.index and not np.isnan(q[drv]):
            out["quali_ratio"][drv] = float(q[drv] * quali_ratio)
        fl = s.laps["lap_time_seconds"].min()
        if pd.notna(fl):
            out["fastest_lap"][drv] = float(fl)
        ob = oracle_base(s)
        if not np.isnan(ob):
            out["oracle"][drv] = ob
    return out


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def eval_masks(sim: DriverSim) -> dict:
    """Boolean mask per variant over the laps the driver actually ran."""
    L = sim.laps
    has_time = ~np.isnan(sim.actual)
    racing = (
        has_time
        & (L["lap_number"].to_numpy() > 1)
        & (L["pit_in_lap"].to_numpy() == 0)
        & (L["pit_out_lap"].to_numpy() == 0)
    )
    neutral = L["is_neutralised"].astype(bool).to_numpy()
    return {
        "incl_neutralised": racing,
        "excl_neutralised": racing & ~neutral,
        "clean": sim.clean_mask & has_time,
    }


def stint_errors(race: RaceData, sims: dict, bases: dict) -> pd.DataFrame:
    """One row per (driver, stint, method, variant) with >= MIN_EVAL_LAPS evaluated laps."""
    rows = []
    for drv, s in sims.items():
        n = len(s.actual)
        masks = eval_masks(s)
        for method in BASE_METHODS:
            if drv not in bases[method]:
                continue
            err = s.component[:n] + bases[method][drv] - s.actual
            for variant, mask in masks.items():
                for k in np.unique(s.stint_index):
                    m = mask & (s.stint_index == k)
                    if m.sum() < MIN_EVAL_LAPS:
                        continue
                    st = s.strategy[k] if k < len(s.strategy) else None
                    rows.append(
                        {
                            "race": race.gp,
                            "driver": drv,
                            "stint": int(k) + 1,
                            "compound": st.compound if st else None,
                            "start_age": st.start_age if st else None,
                            "method": method,
                            "variant": variant,
                            "n_laps": int(m.sum()),
                            "err": float(err[m].mean()),
                        }
                    )
    return pd.DataFrame(rows)


def summarise_stint_errors(df: pd.DataFrame) -> dict:
    """bias / MAE / median |err| / RMSE / relative MAE for one (method, variant) slice."""
    if df.empty:
        return {"n_stints": 0}
    e = df["err"]
    rel = df.groupby(["race", "driver"])["err"].transform(lambda x: x - x.mean())
    multi = df.groupby(["race", "driver"])["err"].transform("size") >= 2
    return {
        "n_stints": int(len(df)),
        "n_laps": int(df["n_laps"].sum()),
        "bias_s_per_lap": round(float(e.mean()), 4),
        "mae_s_per_lap": round(float(e.abs().mean()), 4),
        "median_abs_s_per_lap": round(float(e.abs().median()), 4),
        "rmse_s_per_lap": round(float(np.sqrt((e**2).mean())), 4),
        "relative_mae_s_per_lap": round(float(rel[multi].abs().mean()), 4) if multi.any() else None,
        "n_stints_relative": int(multi.sum()),
    }


def spearman(a, b) -> float:
    a = pd.Series(np.asarray(a, dtype=float)).rank()
    b = pd.Series(np.asarray(b, dtype=float)).rank()
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def classified(race: RaceData, sims: dict) -> pd.DataFrame:
    r = race.results
    r = r[r["status"].isin(CLASSIFIED_STATUSES) & r["driver"].isin(list(sims))]
    return r[["driver", "team", "grid_position", "position", "status"]].copy()


def finishing_order(race: RaceData, sims: dict, bases: dict) -> dict:
    cl = classified(race, sims)
    out = {"n_drivers": int(len(cl))}
    grid = cl["grid_position"].where(cl["grid_position"] > 0, 99)  # 0 = pit-lane start -> back
    out["grid_rho"] = round(spearman(grid, cl["position"]), 4)
    # qualifying order (pre-race pace, no strategy, no grid penalties): what the
    # sim's base_laptime ranking alone would give, so (sim - quali_order) is
    # what the strategy/tyre model adds on top of knowing who is quick.
    qb = race.quali.set_index("driver")["q_best"].reindex(cl["driver"])
    out["quali_order_rho"] = round(spearman(qb.to_numpy(), cl["position"]), 4) if qb.notna().all() else None
    for method in BASE_METHODS:
        ok = cl[cl["driver"].isin(list(bases[method]))]
        totals = [sims[d].sim_total(bases[method][d]) for d in ok["driver"]]
        out[f"sim_rho_{method}"] = round(spearman(totals, ok["position"]), 4) if len(ok) > 2 else None
        out[f"n_{method}"] = int(len(ok))
    return out


def strategy_key(strategy: list) -> tuple:
    return tuple(s.compound for s in strategy)


def stop_laps(strategy: list) -> list[int]:
    return list(np.cumsum([s.laps for s in strategy])[:-1] + 1)


def strategy_pairs(race: RaceData, sims: dict, bases: dict, similar_pct: float = SIMILAR_PACE_PCT) -> pd.DataFrame:
    """
    Similar-pace pairs on different strategies. Strategies are compared as
    actually run (not the full-distance extension) -- every classified
    finisher's last stint only differs by the laps they were lapped.
    """
    cl = classified(race, sims).set_index("driver")
    q = race.quali.set_index("driver")
    base = bases[PRIMARY_METHOD]
    drivers = [d for d in cl.index if d in base and q.at[d, "q_source"] == "own"]
    rows = []
    for i, a in enumerate(drivers):
        for b in drivers[i + 1 :]:
            qa, qb = q.at[a, "q_best"], q.at[b, "q_best"]
            if abs(qa - qb) > similar_pct * min(qa, qb):
                continue
            sa, sb = reconstruct_strategy(sims[a].laps), reconstruct_strategy(sims[b].laps)
            ka, kb = strategy_key(sa), strategy_key(sb)
            if ka != kb:
                kind = "different_sequence"
            else:
                diff = max(abs(x - y) for x, y in zip(stop_laps(sa), stop_laps(sb))) if len(sa) > 1 else 0
                if diff < TIMING_DIFF_LAPS:
                    continue
                kind = "timing_only"
            actual_a_ahead = cl.at[a, "position"] < cl.at[b, "position"]
            sim_diff = sims[a].sim_total(base[a]) - sims[b].sim_total(base[b])
            mean_base = (base[a] + base[b]) / 2
            strat_diff = sims[a].sim_total(mean_base) - sims[b].sim_total(mean_base)
            ga, gb = cl.at[a, "grid_position"], cl.at[b, "grid_position"]
            ga, gb = (ga if ga > 0 else 99), (gb if gb > 0 else 99)
            rows.append(
                {
                    "race": race.gp,
                    "a": a,
                    "b": b,
                    "kind": kind,
                    "strategy_a": "-".join(s.compound[0] + str(s.laps) for s in sa),
                    "strategy_b": "-".join(s.compound[0] + str(s.laps) for s in sb),
                    "quali_gap_pct": round(100 * (qa - qb) / min(qa, qb), 3),
                    "actual_a_ahead": bool(actual_a_ahead),
                    "sim_diff_s": round(float(sim_diff), 2),
                    "strategy_only_diff_s": round(float(strat_diff), 2),
                    "sim_correct": _score(sim_diff, actual_a_ahead),
                    "strategy_only_correct": _score(strat_diff, actual_a_ahead),
                    "grid_correct": _score(ga - gb, actual_a_ahead),
                }
            )
    return pd.DataFrame(rows)


def _score(a_minus_b: float, actual_a_ahead: bool) -> float:
    """1 if 'a ahead iff a_minus_b < 0' matches reality, 0 if not, 0.5 on a tie."""
    if a_minus_b == 0:
        return 0.5
    return float((a_minus_b < 0) == actual_a_ahead)


def wilson_ci(k: float, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (None, None)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (round(float(centre - half), 4), round(float(centre + half), 4))


def race_bootstrap_ci(pairs: pd.DataFrame, col: str, reps: int = BOOTSTRAP_REPS, seed: int = BOOTSTRAP_SEED) -> tuple:
    """95% CI of accuracy, resampling whole races (pairs within a race share drivers)."""
    if pairs.empty:
        return (None, None)
    groups = [g[col].to_numpy() for _, g in pairs.groupby("race")]
    rng = np.random.default_rng(seed)
    accs = []
    for _ in range(reps):
        pick = rng.integers(0, len(groups), len(groups))
        vals = np.concatenate([groups[i] for i in pick])
        accs.append(vals.mean())
    return (round(float(np.percentile(accs, 2.5)), 4), round(float(np.percentile(accs, 97.5)), 4))


def bootstrap_mean_ci(values, reps: int = BOOTSTRAP_REPS, seed: int = BOOTSTRAP_SEED) -> tuple:
    """95% percentile CI of the mean of a small sample (here: one value per race)."""
    v = np.asarray([x for x in values if x is not None and not np.isnan(x)], dtype=float)
    if len(v) == 0:
        return (None, None)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (reps, len(v)))].mean(axis=1)
    return (round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4))


def paired_race_diff_ci(pairs: pd.DataFrame, col_a: str, col_b: str, reps: int = BOOTSTRAP_REPS, seed: int = BOOTSTRAP_SEED) -> dict:
    """Accuracy(col_a) - accuracy(col_b) on the same pairs, race-cluster bootstrap 95% CI."""
    if pairs.empty:
        return {"diff": None, "race_bootstrap95": (None, None)}
    groups = [(g[col_a].to_numpy() - g[col_b].to_numpy()) for _, g in pairs.groupby("race")]
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(reps):
        pick = rng.integers(0, len(groups), len(groups))
        diffs.append(np.concatenate([groups[i] for i in pick]).mean())
    return {
        "diff": round(float((pairs[col_a] - pairs[col_b]).mean()), 4),
        "race_bootstrap95": (round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)),
    }


def summarise_pairs(pairs: pd.DataFrame) -> dict:
    out = {"n_pairs": int(len(pairs)), "n_races": int(pairs["race"].nunique()) if len(pairs) else 0}
    for col in ("sim_correct", "strategy_only_correct", "grid_correct"):
        if pairs.empty:
            out[col.replace("_correct", "")] = None
            continue
        k = float(pairs[col].sum())
        out[col.replace("_correct", "")] = {
            "accuracy": round(k / len(pairs), 4),
            "wilson95": wilson_ci(k, len(pairs)),
            "race_bootstrap95": race_bootstrap_ci(pairs, col),
        }
    out["sim_minus_grid"] = paired_race_diff_ci(pairs, "sim_correct", "grid_correct")
    out["sim_minus_strategy_only"] = paired_race_diff_ci(pairs, "sim_correct", "strategy_only_correct")
    return out


def race_totals(race: RaceData, sims: dict, bases: dict) -> pd.DataFrame:
    """Finished drivers: sim vs real total, all laps and with SC/VSC/red laps removed from both."""
    fin = race.results[race.results["status"] == "Finished"]["driver"]
    rows = []
    for drv in fin:
        if drv not in sims:
            continue
        s = sims[drv]
        if len(s.actual) != len(s.component):
            continue
        actual = actual_race_time(s.laps)
        racing = ~s.laps["is_neutralised"].astype(bool).to_numpy() & ~np.isnan(s.laps["lap_time_seconds"].to_numpy(dtype=float))
        for method in BASE_METHODS:
            if drv not in bases[method]:
                continue
            sim_laps = s.component + bases[method][drv]
            err = sim_laps.sum() - actual
            act_r = s.laps["lap_time_seconds"].to_numpy(dtype=float)[racing].sum()
            err_r = sim_laps[racing].sum() - act_r
            rows.append(
                {
                    "race": race.gp,
                    "driver": drv,
                    "method": method,
                    "pct_error": 100 * err / actual,
                    "pct_error_excl_neutralised": 100 * err_r / act_r,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def run_scorecard(
    gps: Optional[list] = None,
    processed_dir: Path = PROCESSED_DIR,
    raw_dir: Path = JOLPICA_RAW_DIR,
    model_kwargs: Optional[dict] = None,
    extra_train_laps: Optional[pd.DataFrame] = None,
) -> dict:
    """
    model_kwargs: extra keyword arguments for tire_model.build_tire_model (T-020
        experiments: constrain=False, fitted_offsets=False, eras=..., era_weights=...).
        Default None = the shipped defaults.
    extra_train_laps: laps added to EVERY race's LORO training set but never scored
        (e.g. the 2025 staging races when eras includes 2025_regs).
    """
    model_kwargs = model_kwargs or {}
    gps = gps or race_names(processed_dir)
    races = {gp: load_race(gp, processed_dir, raw_dir) for gp in gps}
    all_laps = pd.concat([r.laps for r in races.values()], ignore_index=True)

    stint_frames, pair_frames, total_frames = [], [], []
    per_race, base_info, skipped_all = {}, {}, {}
    for gp, race in races.items():
        # LORO: the model for `gp` never sees its laps
        train_laps = all_laps[all_laps["race_file"] != race.laps["race_file"].iloc[0]]
        if extra_train_laps is not None:
            train_laps = pd.concat([train_laps, extra_train_laps], ignore_index=True)
        model = tm.build_tire_model(laps=train_laps, **model_kwargs)
        sims, skipped = build_driver_sims(race, model)
        train = [(r, build_driver_sims(r, model)[0]) for g, r in races.items() if g != gp]
        ratio = fit_quali_ratio(train) if len(race.quali) else None
        bases = base_laptimes(race, sims, ratio["ratio"] if ratio else None)

        stint_frames.append(stint_errors(race, sims, bases))
        pairs = strategy_pairs(race, sims, bases)
        pair_frames.append(pairs)
        total_frames.append(race_totals(race, sims, bases))

        q_sources = race.quali["q_source"].value_counts().to_dict()
        base_info[gp] = {
            "quali_ratio": round(ratio["ratio"], 5) if ratio else None,
            "quali_ratio_n_train": ratio["n"] if ratio else 0,
            "in_race_oracle_ratio_median": _in_race_ratio(race, sims),
            "quali_sources": {k: int(v) for k, v in q_sources.items()},
            "quali_fallback_drivers": race.quali[race.quali["q_source"] != "own"]["driver"].tolist(),
            "loro_model": {
                "n_laps_fitted": model["_n_laps_fitted"],
                "fuel_coef": round(model["_fuel_coef"], 5),
                **{c: round(model[c]["age_coef"], 5) for c in tm.DRY_COMPOUNDS if c in model},
            },
        }
        skipped_all[gp] = skipped
        per_race[gp] = {"finishing_order": finishing_order(race, sims, bases), "n_pairs": int(len(pairs))}

    stints = pd.concat(stint_frames, ignore_index=True)
    pairs = pd.concat(pair_frames, ignore_index=True)
    totals = pd.concat(total_frames, ignore_index=True)
    return _assemble(gps, stints, pairs, totals, per_race, base_info, skipped_all)


def _in_race_ratio(race: RaceData, sims: dict) -> Optional[float]:
    """Diagnostic (leaky): this race's own median oracle/quali ratio -- what the LORO ratio is trying to guess."""
    q = race.quali.set_index("driver")
    r = [oracle_base(s) / q.at[d, "q_best"] for d, s in sims.items() if d in q.index and q.at[d, "q_source"] == "own"]
    r = [x for x in r if not np.isnan(x)]
    return round(float(np.median(r)), 5) if r else None


def _assemble(gps, stints, pairs, totals, per_race, base_info, skipped) -> dict:
    stint_summary = {
        m: {v: summarise_stint_errors(stints[(stints.method == m) & (stints.variant == v)]) for v in VARIANTS}
        for m in BASE_METHODS
    }
    by_compound = {
        m: {
            v: {
                c: summarise_stint_errors(g)
                for c, g in stints[(stints.method == m) & (stints.variant == v)].groupby("compound")
            }
            for v in VARIANTS
        }
        for m in BASE_METHODS
    }
    by_race = {
        gp: summarise_stint_errors(
            stints[(stints.method == PRIMARY_METHOD) & (stints.variant == HEADLINE_VARIANT) & (stints.race == gp)]
        )
        for gp in gps
    }
    fo = pd.DataFrame({gp: d["finishing_order"] for gp, d in per_race.items()}).T
    fo_summary = {
        "grid_rho_mean": round(float(fo["grid_rho"].astype(float).mean()), 4),
        **{
            f"sim_rho_mean_{m}": round(float(fo[f"sim_rho_{m}"].astype(float).mean()), 4)
            for m in BASE_METHODS
        },
        "quali_order_rho_mean": round(float(fo["quali_order_rho"].astype(float).mean()), 4),
        "sim_minus_grid_mean_race_bootstrap95": bootstrap_mean_ci(
            (fo[f"sim_rho_{PRIMARY_METHOD}"].astype(float) - fo["grid_rho"].astype(float)).tolist()
        ),
        "n_races": int(len(fo)),
        "n_drivers": int(fo["n_drivers"].sum()),
        f"races_sim_beats_grid_{PRIMARY_METHOD}": int((fo[f"sim_rho_{PRIMARY_METHOD}"].astype(float) > fo["grid_rho"].astype(float)).sum()),
    }
    primary_pairs = pairs[pairs["kind"] == "different_sequence"] if len(pairs) else pairs
    tot_summary = {
        m: {
            "n_drivers": int((totals.method == m).sum()),
            "mean_abs_pct": round(float(totals[totals.method == m]["pct_error"].abs().mean()), 3),
            "mean_pct": round(float(totals[totals.method == m]["pct_error"].mean()), 3),
            "mean_abs_pct_excl_neutralised": round(float(totals[totals.method == m]["pct_error_excl_neutralised"].abs().mean()), 3),
            "mean_pct_excl_neutralised": round(float(totals[totals.method == m]["pct_error_excl_neutralised"].mean()), 3),
        }
        for m in BASE_METHODS
    }
    h_primary = stint_summary[PRIMARY_METHOD][HEADLINE_VARIANT]
    leakage_cost = {
        "definition": (
            "non-leaky quali_ratio minus the leaky reference; positive MAE/error deltas = what not "
            "leaking costs. oracle = base fitted on the scored race's own clean laps (floor)."
        ),
        "per_stint_mae_excl_neutralised": {
            "quali_ratio": h_primary["mae_s_per_lap"],
            **{m: stint_summary[m][HEADLINE_VARIANT]["mae_s_per_lap"] for m in LEAKY_METHODS},
            **{
                f"cost_vs_{m}": round(h_primary["mae_s_per_lap"] - stint_summary[m][HEADLINE_VARIANT]["mae_s_per_lap"], 4)
                for m in LEAKY_METHODS
            },
        },
        "finishing_order_rho_mean": {
            PRIMARY_METHOD: fo_summary[f"sim_rho_mean_{PRIMARY_METHOD}"],
            **{m: fo_summary[f"sim_rho_mean_{m}"] for m in LEAKY_METHODS},
        },
        "race_total_mean_abs_pct_excl_neutralised": {m: tot_summary[m]["mean_abs_pct_excl_neutralised"] for m in BASE_METHODS},
    }
    return {
        "meta": {
            "scorecard_version": SCORECARD_VERSION,
            "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "git_head": _git_head(),
            "races": gps,
            "primary_base_method": PRIMARY_METHOD,
            "leaky_methods_reference_only": list(LEAKY_METHODS),
            "headline_variant": HEADLINE_VARIANT,
            "config": {
                "min_eval_laps_per_stint": MIN_EVAL_LAPS,
                "similar_pace_pct": SIMILAR_PACE_PCT,
                "timing_diff_laps": TIMING_DIFF_LAPS,
                "bootstrap_reps": BOOTSTRAP_REPS,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "pit_loss_s": DEFAULT_PIT_LOSS_SECONDS,
                "tire_model": "build_tire_model(laps=other 9 races) defaults (linear wear)",
                "noise": "none (deterministic engine)",
            },
            "skipped_drivers": skipped,
        },
        "headline": {
            "per_stint_mae_s_per_lap": stint_summary[PRIMARY_METHOD][HEADLINE_VARIANT].get("mae_s_per_lap"),
            "per_stint_bias_s_per_lap": stint_summary[PRIMARY_METHOD][HEADLINE_VARIANT].get("bias_s_per_lap"),
            "per_stint_relative_mae_s_per_lap": stint_summary[PRIMARY_METHOD][HEADLINE_VARIANT].get("relative_mae_s_per_lap"),
            "per_stint_mae_incl_neutralised": stint_summary[PRIMARY_METHOD]["incl_neutralised"].get("mae_s_per_lap"),
            "finishing_order_sim_rho": fo_summary[f"sim_rho_mean_{PRIMARY_METHOD}"],
            "finishing_order_grid_rho": fo_summary["grid_rho_mean"],
            "strategy_ranking_accuracy": (summarise_pairs(primary_pairs).get("sim") or {}).get("accuracy"),
            "strategy_ranking_n_pairs": int(len(primary_pairs)),
        },
        "base_laptime": base_info,
        "leakage_cost": leakage_cost,
        "per_stint": {"summary": stint_summary, "by_compound": by_compound, f"by_race_{PRIMARY_METHOD}_{HEADLINE_VARIANT}": by_race},
        "finishing_order": {"summary": fo_summary, "per_race": {gp: d["finishing_order"] for gp, d in per_race.items()}},
        "strategy_ranking": {
            "definition": (
                f"classified finishers, own quali times within {100 * SIMILAR_PACE_PCT:.1f}% of each other, "
                "different compound sequence / stop count; broad also admits same sequence with a stop "
                f">= {TIMING_DIFF_LAPS} laps apart"
            ),
            "primary": summarise_pairs(primary_pairs),
            "broad": summarise_pairs(pairs),
            "pairs": pairs.to_dict(orient="records"),
        },
        "race_totals": tot_summary,
    }


def _git_head() -> Optional[str]:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10
        ).stdout.strip() or None
    except Exception:
        return None


# --------------------------------------------------------------------------
# printing
# --------------------------------------------------------------------------


def _fmt(x, nd=3):
    return "  n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.{nd}f}" if nd else str(x)


def print_scorecard(card: dict) -> None:
    h = card["headline"]
    print("=" * 78)
    print(f"VALIDATION SCORECARD v{card['meta']['scorecard_version']}  ({len(card['meta']['races'])} races, LORO, "
          f"base_laptime={card['meta']['primary_base_method']})")
    print("=" * 78)
    print(f"Per-stint MAE (excl SC/VSC/red): {h['per_stint_mae_s_per_lap']:.3f} s/lap   bias {h['per_stint_bias_s_per_lap']:+.3f}"
          f"   relative MAE {h['per_stint_relative_mae_s_per_lap']:.3f}")
    print(f"Per-stint MAE (incl SC/VSC/red): {h['per_stint_mae_incl_neutralised']:.3f} s/lap")
    print(f"Finishing-order Spearman: sim {h['finishing_order_sim_rho']:.3f}  vs grid {h['finishing_order_grid_rho']:.3f}")
    sr = card["strategy_ranking"]["primary"]
    if sr["n_pairs"]:
        print(f"Strategy-ranking accuracy: {sr['sim']['accuracy']:.3f}  (n={sr['n_pairs']} pairs, {sr['n_races']} races; "
              f"race-bootstrap 95% CI {sr['sim']['race_bootstrap95']})")

    print("\n-- base_laptime per held-out race --")
    print(f"{'race':<10} {'LORO ratio':>10} {'in-race':>8} {'fallbacks'}")
    for gp, b in card["base_laptime"].items():
        print(f"{gp:<10} {b['quali_ratio'] or float('nan'):>10.4f} {b['in_race_oracle_ratio_median'] or float('nan'):>8.4f} "
              f"{','.join(b['quali_fallback_drivers']) or '-'}")

    print("\n-- per-stint error, s/lap (MAE / bias / relative MAE; n stints) --")
    print(f"{'method':<22}" + "".join(f"{v:>27}" for v in VARIANTS))
    for m, by_v in card["per_stint"]["summary"].items():
        label = m + (" (LEAKY)" if m in LEAKY_METHODS else "")
        cells = []
        for v in VARIANTS:
            s = by_v[v]
            cells.append(f"{s['mae_s_per_lap']:.3f}/{s['bias_s_per_lap']:+.3f}/{s['relative_mae_s_per_lap']:.3f} n={s['n_stints']}")
        print(f"{label:<22}" + "".join(f"{c:>27}" for c in cells))
    print(f"\n   median |err| ({PRIMARY_METHOD}): " + ", ".join(
        f"{v} {card['per_stint']['summary'][PRIMARY_METHOD][v]['median_abs_s_per_lap']:.3f}" for v in VARIANTS))

    lc = card["leakage_cost"]["per_stint_mae_excl_neutralised"]
    print(f"\n   cost of not leaking (MAE excl SC/VSC/red): quali_ratio {lc['quali_ratio']:.3f} vs fastest_lap "
          f"{lc['fastest_lap']:.3f} ({lc['cost_vs_fastest_lap']:+.3f}) vs oracle floor {lc['oracle']:.3f} ({lc['cost_vs_oracle']:+.3f})")

    for variant in ("incl_neutralised", HEADLINE_VARIANT):
        print(f"\n-- per-stint by compound ({PRIMARY_METHOD}, {variant}) --")
        for c, s in card["per_stint"]["by_compound"][PRIMARY_METHOD][variant].items():
            rel = s["relative_mae_s_per_lap"]
            print(f"  {c:<12} n={s['n_stints']:>3}  MAE {s['mae_s_per_lap']:.3f}  median {s['median_abs_s_per_lap']:.3f}  "
                  f"bias {s['bias_s_per_lap']:+.3f}  relative MAE {rel if rel is not None else float('nan'):.3f}")
    print(f"  (oracle base, same variant -- shape-only floor):")
    for c, s in card["per_stint"]["by_compound"]["oracle"][HEADLINE_VARIANT].items():
        print(f"  {c:<7} n={s['n_stints']:>3}  MAE {s['mae_s_per_lap']:.3f}  bias {s['bias_s_per_lap']:+.3f}")

    print(f"\n-- per-stint by race ({PRIMARY_METHOD}, {HEADLINE_VARIANT}) --")
    for gp, s in card["per_stint"][f"by_race_{PRIMARY_METHOD}_{HEADLINE_VARIANT}"].items():
        print(f"  {gp:<10} n={s['n_stints']:>3}  MAE {s['mae_s_per_lap']:.3f}  bias {s['bias_s_per_lap']:+.3f}  "
              f"relative MAE {s['relative_mae_s_per_lap']:.3f}")

    print("\n-- finishing order: Spearman rho vs results.csv position --")
    print(f"{'race':<10} {'n':>3} {'grid':>7} {'quali_ord':>9} " + " ".join(f"{m:>12}" for m in BASE_METHODS))
    for gp, f in card["finishing_order"]["per_race"].items():
        print(f"{gp:<10} {f['n_drivers']:>3} {f['grid_rho']:>7.3f} {f['quali_order_rho'] or float('nan'):>9.3f} " + " ".join(
            f"{(f[f'sim_rho_{m}'] if f[f'sim_rho_{m}'] is not None else float('nan')):>12.3f}" for m in BASE_METHODS))
    fs = card["finishing_order"]["summary"]
    print(f"{'mean':<10} {fs['n_drivers']:>3} {fs['grid_rho_mean']:>7.3f} {fs['quali_order_rho_mean']:>9.3f} " + " ".join(
        f"{fs[f'sim_rho_mean_{m}']:>12.3f}" for m in BASE_METHODS))
    print(f"  one-car sim ({PRIMARY_METHOD}) minus grid: mean {fs[f'sim_rho_mean_{PRIMARY_METHOD}'] - fs['grid_rho_mean']:+.3f}, "
          f"race-bootstrap 95% CI {fs['sim_minus_grid_mean_race_bootstrap95']}, beats grid in "
          f"{fs[f'races_sim_beats_grid_{PRIMARY_METHOD}']}/{fs['n_races']} races")

    print("\n-- strategy ranking (similar-pace pairs on different strategies) --")
    for name in ("primary", "broad"):
        s = card["strategy_ranking"][name]
        if not s["n_pairs"]:
            print(f"  {name}: no pairs")
            continue
        print(f"  {name}: n={s['n_pairs']} pairs over {s['n_races']} races")
        for col in ("sim", "strategy_only", "grid"):
            r = s[col]
            print(f"    {col:<14} acc {r['accuracy']:.3f}  Wilson95 {r['wilson95']}  race-bootstrap95 {r['race_bootstrap95']}")
        for k in ("sim_minus_grid", "sim_minus_strategy_only"):
            print(f"    {k:<24} {s[k]['diff']:+.3f}  race-bootstrap95 {s[k]['race_bootstrap95']}")

    print("\n-- race totals, Finished drivers (mean |%| / mean %; all laps | excl SC/VSC/red) --")
    for m, t in card["race_totals"].items():
        label = m + (" (LEAKY)" if m in LEAKY_METHODS else "")
        print(f"  {label:<22} n={t['n_drivers']:>3}  {t['mean_abs_pct']:.2f}% / {t['mean_pct']:+.2f}%  |  "
              f"{t['mean_abs_pct_excl_neutralised']:.2f}% / {t['mean_pct_excl_neutralised']:+.2f}%")

    sk = {gp: s for gp, s in card["meta"]["skipped_drivers"].items() if s}
    if sk:
        print("\nskipped (strategy not simulatable): " + "; ".join(f"{gp}: {', '.join(s)}" for gp, s in sk.items()))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def write_scorecard(card: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(card, fh, indent=2, default=_json_default)
        fh.write("\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    card = run_scorecard()
    print_scorecard(card)
    if not args.no_write:
        write_scorecard(card, args.out)
        print(f"\nwrote {args.out}")
