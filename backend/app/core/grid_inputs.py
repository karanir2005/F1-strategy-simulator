"""
Inputs for the full-grid simulation (T-034): who is on the grid, how fast each car is, how likely it is
to retire, and which strategy it runs. Kept apart from race_grid_sim.py so the engine stays pure numpy.

Data path: the same cached Jolpica tables the predictor uses (data/jolpica/derived/*.csv, read through
predictor.prepare_history), so nothing here touches the network. Every number for race (season, rnd) is
built from rounds strictly before it, except the target race's own qualifying times and starting grid
(known before the start, the same two things the predictor's grid mode reads).

Pace (seconds per lap slower than the field's best cars), per car:
  1. Qualifying gap: Q1 time / fastest Q1 time - 1. Q1 is the one session every car sets, so the gap is
     comparable across the whole grid (Q3 laps are on a faster track and are not).
  2. Shrunk toward the team mean: a driver's quali gap leans on his team-mate's (DRIVER_SHRINK).
  3. Optionally blended with the predictor's race-rating (Plackett-Luce theta_race from earlier races of
     the season) by quantile-matching: the rating order is mapped onto the sorted quali gaps of this
     weekend, so the rating contributes ordering information without needing its own unit conversion.
  4. Converted to race seconds with RACE_PACE_SCALE x base lap time. Practice long runs (openf1_practice)
     are not used: they are only available for part of the 2026 weekends and the predictor's own test
     of them (use_fp) was not a clear gain.
The pace constants below were T-034 PLACEHOLDERs; T-060 fits them (calibrate_grid_sim.py, leave-one-race-out against
real race pace) into data/sim_params/grid_sim_calibration.json and they are read from there at import, with the old
placeholder values as the per-key fallback (race_grid_sim.load_grid_calibration). Callers that need other values
(the validation does it per fold) pass them explicitly.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.core import predictor as P
from app.core.optimizer import NoFeasiblePlan, min_stint_laps, optimize
from app.core.race_grid_sim import CarSpec, load_grid_calibration
from app.core.race_simulator import DEFAULT_PIT_LOSS_SECONDS, Stint
from app.core.track_params import get_track

REPO_ROOT = Path(__file__).resolve().parents[3]
DERIVED = REPO_ROOT / "data" / "jolpica" / "derived"
SEASON = 2026

_CAL = load_grid_calibration()
# Race-pace spread as a share of the Q1 spread (T-060: fitted, the placeholder 0.6 was too low: real race pace is
# spread MORE than the Q1 gaps scaled by 1 in 2026; see grid_sim_calibration.json). Falls back to 0.6.
RACE_PACE_SCALE = _CAL.race_pace_scale
# Weight of a driver's own quali gap vs the team mean (1 = no shrinkage). Fitted (T-060), fallback 0.5.
DRIVER_SHRINK = _CAL.driver_shrink
# Weight of the quali-based pace vs the earlier-race rating-based pace (1 = quali only). Fitted (T-060), fallback 0.5.
QUALI_WEIGHT = _CAL.quali_weight
# Cap on the Q1 gap (percent of lap) so a car with a crashed/no lap does not get an absurd pace.
MAX_GAP_FRAC = 0.04

# Jolpica circuit_id -> track_params.json track_id (2026 calendar).
CIRCUIT_TO_TRACK = {
    "albert_park": "australia", "shanghai": "china", "suzuka": "japan", "miami": "miami",
    "villeneuve": "canada", "monaco": "monaco", "catalunya": "barcelona", "red_bull_ring": "austria",
    "silverstone": "britain", "spa": "belgium", "hungaroring": "hungary", "zandvoort": "netherlands",
    "monza": "italy", "madring": "madrid", "baku": "azerbaijan", "sepang": "malaysia",
    "marina_bay": "singapore", "americas": "austin", "rodriguez": "mexico", "interlagos": "brazil",
    "vegas": "las_vegas", "losail": "qatar", "yas_marina": "abudhabi",
}
TRACK_TO_CIRCUIT = {v: k for k, v in CIRCUIT_TO_TRACK.items()}


@lru_cache(maxsize=1)
def load_history() -> dict:
    """Predictor history dict from the cached derived Jolpica tables (offline)."""
    res = pd.read_csv(DERIVED / "results.csv")
    qual = pd.read_csv(DERIVED / "qualifying.csv")
    sched = pd.read_csv(DERIVED / "schedule.csv")
    spr = pd.read_csv(DERIVED / "sprint.csv") if (DERIVED / "sprint.csv").exists() else None
    tr_path = DERIVED / "results_2018_2021.csv"
    tr = pd.read_csv(tr_path) if tr_path.exists() else None
    return P.prepare_history(res, qual, sprint=spr, track_results=tr, schedule=sched)


def round_for_track(hist: dict, track_id: str, season: int = SEASON) -> Optional[int]:
    cid = TRACK_TO_CIRCUIT.get(track_id)
    sc = hist["schedule"]
    r = sc[(sc["season"] == season) & (sc["circuit_id"] == cid)]
    return int(r["round"].iloc[0]) if len(r) else None


def _lap_seconds(s) -> float:
    if s is None or (isinstance(s, float) and np.isnan(s)) or str(s).strip() == "":
        return np.nan
    s = str(s)
    if ":" in s:
        m, sec = s.split(":")
        return 60.0 * float(m) + float(sec)
    return float(s)


def race_entrants(hist: dict, season: int, rnd: int, grid: Optional[Dict[str, int]] = None) -> pd.DataFrame:
    """
    Entry list with starting slots: columns driver_id, code, team_id, team, grid (1..n, pit-lane starters last).
    Source of the slot, in order: `grid` {code: slot} if given; the race's own results grid (race already
    run); else the qualifying classification (no penalties applied), with no-time drivers at the back.
    """
    res = hist["results"]
    r = res[(res["season"] == season) & (res["round"] == rnd)]
    q = hist["qualifying"]
    q = q[(q["season"] == season) & (q["round"] == rnd)]
    cols = ["driver_id", "code", "team_id", "team"]
    if len(r):
        ent = r[cols + ["grid"]].copy()
    elif len(q):
        ent = P.entrants_for(hist, season, rnd).copy()
        prev = P.entrants_for(hist, season, rnd - 1)
        extra = prev[~prev["driver_id"].isin(ent["driver_id"])]
        ent = pd.concat([ent, extra], ignore_index=True)
        ent = ent.merge(q[["driver_id", "quali_position"]], on="driver_id", how="left")
        ent["grid"] = ent["quali_position"]
        ent = ent.drop(columns="quali_position")
    else:
        raise ValueError(f"no results or qualifying for {season} round {rnd}: pass an explicit grid")
    if grid:
        g = {str(k).upper(): int(v) for k, v in grid.items()}
        unknown = sorted(set(g) - set(ent["code"]))
        if unknown:
            raise ValueError(f"grid names drivers not in the entry list: {unknown}")
        ent["grid"] = ent["code"].map(g).where(ent["code"].isin(g), np.nan)
    # slots: sort by (grid, with 0/NaN = back), then renumber 1..n so slots are distinct and complete
    key = pd.to_numeric(ent["grid"], errors="coerce").fillna(0).to_numpy(dtype=float)
    key[key <= 0] = 1e3
    order = np.argsort(key, kind="stable")
    out = ent.iloc[order].reset_index(drop=True)
    out["grid"] = np.arange(1, len(out) + 1)
    return out


def quali_gaps(hist: dict, season: int, rnd: int, ent: pd.DataFrame) -> np.ndarray:
    """Q1 gap to the fastest Q1 time as a fraction of lap time, per entrant; missing -> the worst gap seen."""
    q = hist["qualifying"]
    q = q[(q["season"] == season) & (q["round"] == rnd)]
    t = ent["driver_id"].map(dict(zip(q["driver_id"], q["q1"].map(_lap_seconds))))
    t = pd.to_numeric(t, errors="coerce")
    if t.notna().sum() < 2:
        return np.full(len(ent), np.nan)
    g = (t / t.min() - 1.0).clip(upper=MAX_GAP_FRAC)
    return g.fillna(g.max()).to_numpy(dtype=float)


def shrink_to_team(gap: np.ndarray, team_ids: pd.Series, lam: float = DRIVER_SHRINK) -> np.ndarray:
    gap = np.asarray(gap, dtype=float)
    team_mean = pd.Series(gap).groupby(team_ids.to_numpy()).transform("mean").to_numpy()
    return team_mean + lam * (gap - team_mean)


def pace_deltas(
    hist: dict,
    season: int,
    rnd: int,
    ent: pd.DataFrame,
    base_laptime_s: float,
    race_pace_scale: float = RACE_PACE_SCALE,
    driver_shrink: float = DRIVER_SHRINK,
    quali_weight: float = QUALI_WEIGHT,
    cfg: Optional[P.PredictorConfig] = None,
) -> np.ndarray:
    """
    Seconds per lap behind the best car (>= 0) for each entrant. See module docstring. If the target
    weekend has no qualifying times (e.g. a grid override for a race before qualifying), the pace comes from
    the ratings alone, and if there are no earlier races either, every car gets the same pace.
    """
    n = len(ent)
    gq = quali_gaps(hist, season, rnd, ent)
    has_q = not np.isnan(gq).all()
    theta = None
    if quali_weight < 1.0 or not has_q:
        try:
            f = P.race_features(hist, season, rnd, cfg or P.PredictorConfig(), entry_list=ent[["driver_id", "code", "team_id", "team"]], grid=ent["grid"].to_numpy(dtype=float))
            theta = f["theta_race"].to_numpy(dtype=float)
            if np.allclose(theta, theta[0]):
                theta = None
        except Exception:  # no earlier race of the season -> no ratings
            theta = None
    if has_q:
        gq_s = shrink_to_team(gq, ent["team_id"], driver_shrink)
        if theta is None or quali_weight >= 1.0:
            gap = gq_s
        else:
            # quantile-match the rating order onto the sorted quali gaps, then blend
            srt = np.sort(gq_s)
            rank = np.empty(n, dtype=int)
            rank[np.argsort(-theta, kind="stable")] = np.arange(n)
            gap = quali_weight * gq_s + (1 - quali_weight) * srt[rank]
    elif theta is not None:
        # no quali times: spread of a typical field (0..1.5% of a lap) assigned by rating order
        srt = np.linspace(0.0, 0.015, n)
        rank = np.empty(n, dtype=int)
        rank[np.argsort(-theta, kind="stable")] = np.arange(n)
        gap = srt[rank]
    else:
        gap = np.zeros(n)
    gap = gap - gap.min()
    return race_pace_scale * gap * base_laptime_s


def dnf_probs(hist: dict, season: int, rnd: int, ent: pd.DataFrame, cfg: Optional[P.PredictorConfig] = None) -> np.ndarray:
    """Predictor's P(not classified) per entrant, grid-slot adjusted (grid mode), from earlier rounds only."""
    e = ent[["driver_id", "team_id"]].copy()
    e["grid_eff"] = P._grid_eff(ent["grid"], len(ent))
    return P.dnf_probabilities(hist, season, rnd, e, cfg or P.PredictorConfig(), use_grid=True)


# ------------------------------------------------------------------ strategies


_PLAN_CACHE: Dict[tuple, list] = {}


def top_plans(track: dict, tire_model: dict, total_laps: Optional[int] = None, n_plans: int = 3, n_sims: int = 400) -> List[List[Stint]]:
    """The optimizer's top plans for the track (rank 1 first), as Stint lists. Cached per (track, laps)."""
    total = int(total_laps or track["total_laps"])
    key = (track["track_id"], total, id(tire_model), n_plans, n_sims)
    if key not in _PLAN_CACHE:
        res = optimize(
            tire_model,
            total_laps=total,
            pit_loss_s=track["pit_loss_s"],
            base_laptime_s=track["base_laptime_s"],
            max_stint_laps=track["max_stint_laps"],
            typical_stint_laps=track.get("typical_stint_laps"),
            max_stops=2,
            n_sims=n_sims,
            top_n=n_plans,
            seed=0,
        )
        _PLAN_CACHE[key] = [[Stint(s["compound"], int(s["laps"]), int(s.get("start_age", 1))) for s in p["stints"]] for p in res.plans]
    return _PLAN_CACHE[key]


PLAN_WEIGHTS = _CAL.plan_weights  # share of cars on the optimizer's #1 / #2 / #3 plan (T-060 fit, fallback 0.6/0.25/0.15)
PIT_JITTER_LAPS = _CAL.pit_jitter_laps  # per-stop +-laps of per-car variation (T-060 fit, fallback 2)


def vary_plan(plan: List[Stint], rng: np.random.Generator, total_laps: int, jitter: Optional[int] = None) -> List[Stint]:
    """Shift every stop by an integer in [-J, J] laps, keeping each stint >= the optimizer's minimum length."""
    J = PIT_JITTER_LAPS if jitter is None else int(jitter)
    if len(plan) < 2:
        return [Stint(s.compound, s.laps, s.start_age) for s in plan]
    lo = min_stint_laps(total_laps)
    bounds = np.cumsum([s.laps for s in plan])[:-1].astype(int)
    shifted = []
    prev = 0
    for k, b in enumerate(bounds):
        nb = int(b + rng.integers(-J, J + 1))
        nb = max(nb, prev + lo)
        nb = min(nb, total_laps - lo * (len(bounds) - k))
        shifted.append(nb)
        prev = nb
    edges = [0] + shifted + [total_laps]
    return [Stint(s.compound, edges[i + 1] - edges[i], s.start_age) for i, s in enumerate(plan)]


def default_strategies(
    n_cars: int, track: dict, tire_model: dict, total_laps: Optional[int] = None, seed: int = 0, vary: bool = True,
    plan_weights=None, pit_jitter: Optional[int] = None,
) -> List[List[Stint]]:
    """One plan per car: the optimizer's top plans assigned by PLAN_WEIGHTS, each with per-stop jitter."""
    total = int(total_laps or track["total_laps"])
    plans = top_plans(track, tire_model, total)
    rng = np.random.default_rng(seed + 7919)
    w = np.array((PLAN_WEIGHTS if plan_weights is None else plan_weights)[: len(plans)], dtype=float)
    w = w / w.sum()
    out = []
    for _ in range(n_cars):
        base = plans[int(rng.choice(len(plans), p=w))] if vary else plans[0]
        out.append(vary_plan(base, rng, total, pit_jitter) if vary else [Stint(s.compound, s.laps, s.start_age) for s in base])
    return out


@dataclass
class GridInputs:
    season: int
    rnd: int
    track: dict
    total_laps: int
    cars: List[CarSpec]
    ent: pd.DataFrame
    notes: List[str]


def build_inputs(
    track_id: str,
    tire_model: dict,
    season: int = SEASON,
    rnd: Optional[int] = None,
    grid: Optional[Dict[str, int]] = None,
    strategies: Optional[Dict[str, List[Stint]]] = None,
    total_laps: Optional[int] = None,
    seed: int = 0,
    hist: Optional[dict] = None,
    base_laptime_s: Optional[float] = None,
    race_pace_scale: float = RACE_PACE_SCALE,
    quali_weight: float = QUALI_WEIGHT,
    vary_strategies: bool = True,
    driver_shrink: Optional[float] = None,
    plan_weights=None,
    pit_jitter: Optional[int] = None,
    track: Optional[dict] = None,
) -> GridInputs:
    """Assemble CarSpecs for a race. `strategies` overrides {driver code: [Stint]} for those cars only."""
    hist = hist or load_history()
    track = track if track is not None else get_track(track_id)  # `track`: caller-built entry (validation rebuilds it per fold, T-060)
    if track is None:
        raise KeyError(track_id)
    if rnd is None:
        rnd = round_for_track(hist, track_id, season)
        if rnd is None:
            raise ValueError(f"track {track_id!r} is not on the {season} calendar; pass round explicitly")
    total = int(total_laps or track["total_laps"])
    base = float(base_laptime_s or track["base_laptime_s"])
    ent = race_entrants(hist, season, rnd, grid)
    pace = pace_deltas(
        hist, season, rnd, ent, base, race_pace_scale=race_pace_scale, quali_weight=quali_weight,
        driver_shrink=DRIVER_SHRINK if driver_shrink is None else driver_shrink,
    )
    pdnf = dnf_probs(hist, season, rnd, ent)
    strats = default_strategies(len(ent), track, tire_model, total, seed, vary_strategies, plan_weights, pit_jitter)
    notes = []
    overrides = {str(k).upper(): v for k, v in (strategies or {}).items()}
    unknown = sorted(set(overrides) - set(ent["code"]))
    if unknown:
        raise ValueError(f"strategy override for drivers not on the grid: {unknown}")
    cars = []
    for i, r in ent.iterrows():
        strat = overrides.get(r["code"], strats[i])
        cars.append(CarSpec(r["code"], str(r["team"]), int(r["grid"]), float(pace[i]), strat, float(pdnf[i])))
    if overrides:
        notes.append(f"strategy overridden for: {sorted(overrides)}")
    return GridInputs(season, int(rnd), track, total, cars, ent, notes)
