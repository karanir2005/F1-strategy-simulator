"""
Full-grid multi-car race simulation (T-034, Roadmap Phase 1 + the Phase 2 event layer).

The single-car engine (race_simulator / monte_carlo) prices a strategy in clean air against a fixed
clock. This module runs every car of the grid in the same race, so position, traffic, undercut and
overcut, safety-car bunching and DNFs interact, and it returns a finishing-position distribution per
car instead of a race-time distribution for one.

Model, per lap, vectorised over N Monte Carlo runs (arrays are (N runs, D cars))
--------------------------------------------------------------------------------
State is each car's cumulative race time T. Gaps are differences in T, so a pit stop re-joins "at the
real gap" (T grows by the stop time and the sort puts the car where it lands): undercut and overcut
are not coded anywhere, they fall out of the arithmetic.

  1. Order cars by T. gap_ahead = T - T(car in front).
  2. Free lap time = track base lap + car pace delta + tyre delta (fitted, by compound and age)
     + fuel effect + Gaussian lap noise (calibrated level). Pit loss lands on the out-lap, the same
     convention as race_simulator.simulate_race.
  3. Dirty air: gap_ahead < window  ->  + time_loss and + extra_deg_frac x the tyre delta.
  4. Overtakes, resolved from the front of the order backwards: a car whose free lap would put it
     whose gap to the car in front at the start of the lap is within overtake.window_s attempts a pass
     with the fitted logistic probability of pace delta, tyre-age delta and gap (T-053 definitions).
     Success: attacker pays pass_time_cost.attacker, the defender pays pass_time_cost.defender and ends
     the lap behind. Failure: attacker is held at min_follow_gap. One attempt per car per lap.
  5. Safety car: every car runs sc_lap_multiple x a neutral lap, no overtakes, and gaps close towards
     sc_gap_s (a fraction sc_gap_closure of the extra gap is removed per SC lap), so the field bunches.
     VSC: slower laps, gaps frozen, no overtakes. Stops under SC/VSC cost the fitted pit_loss_ratio.
     The timeline comes from safety_cars.py (fitted) and is shared by every car in a run;
     forced_sc_lap forces an SC from a given lap.
  6. DNFs: each car retires with its P(DNF) at a uniformly random lap; a retired car never finishes
     (its clock gets +BIG per remaining lap, so earlier retirements rank lower).
  7. Lap 1 is not simulated physically: the post-lap-1 order is the grid order perturbed by the fitted
     per-grid-bin position change (mean, sd), with fixed spacing between cars.

Common random numbers: every random quantity is drawn with a shape that does not depend on strategies,
SC settings or interaction parameters (per-lap draws are (N, D) from one seeded Generator, in car
order), so changing one car's strategy or the SC setting reuses the same noise, pace draws and DNFs.
Deterministic given the seed.

NOT modelled (see the T-034 handoff for the list and why): lapping and blue flags, pit-lane queueing
(two cars of one team stopping together), red flags, weather, DRS/overtake-mode as a separate mechanic
(it is inside the fitted overtake logistic), tyre-temperature, fuel-saving, team orders, reactive strategy
(plans are fixed in advance per car), start-procedure penalties, in-race penalties, SC-triggering DNFs,
per-lap correlated noise.

PLACEHOLDER parameters: interaction parameters are fitted by T-053 into data/sim_params/race_interaction.json.
Until that file exists (or for any key it omits) the values in PLACEHOLDER_INTERACTION are used and
InteractionParams.placeholder_keys says which. The other modelling constants below flagged PLACEHOLDER
need a fit as well.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.core.monte_carlo import DEFAULT_LAP_TIME_NOISE_STD, DEFAULT_PIT_LOSS_NOISE_STD
from app.core.race_simulator import DEFAULT_PIT_LOSS_SECONDS, Stint
from app.core.safety_cars import (
    SafetyCarParams,
    apply_forced_sc,
    default_sc_length,
    draw_timelines,
)
from app.core.tire_model import compound_pace_offset, fuel_effect, predict_delta

INTERACTION_PARAMS_PATH = Path(__file__).resolve().parents[3] / "data" / "sim_params" / "race_interaction.json"
CALIBRATION_PATH = Path(__file__).resolve().parents[3] / "data" / "sim_params" / "grid_sim_calibration.json"

# FIA race points P1..P10 (same table as predictor.FIA_POINTS; duplicated so this module does not import
# scipy/pandas through the predictor).
FIA_POINTS = np.array([25, 18, 15, 12, 10, 8, 6, 4, 2, 1] + [0] * 20, dtype=float)

# A retired car's clock gets this many seconds per remaining lap, so every classified car (even a lap-down
# one) sorts ahead of it and an earlier retirement sorts behind a later one.
DNF_BIG_S = 1.0e5

# Grid bins of the lap-1 table, as in the T-053 schema.
LAP1_BINS = (("1-3", 1, 3), ("4-6", 4, 6), ("7-10", 7, 10), ("11-15", 11, 15), ("16-22", 16, 99))

# PLACEHOLDER interaction values, used until data/sim_params/race_interaction.json provides fitted ones.
# Chosen to be plausible, not measured. Sign conventions of the overtake logistic (T-053 must match):
#   logit p = intercept + pace_delta_coef * pace_delta_s + tyre_age_delta_coef * tyre_age_delta
#             + gap_coef * gap_s + track_offset[track_id]
#   pace_delta_s   = defender free lap time - attacker free lap time   (> 0: attacker is faster)
#   tyre_age_delta = defender tyre age - attacker tyre age            (> 0: attacker on fresher tyres)
#   gap_s          = gap to the car in front at the START of the lap
# lap1 mean/sd (T-053 definition): end-of-lap-1 position - grid position (> 0 = LOST places).
PACE_DELTA_CLIP_S = 2.0  # the overtake logistic was fitted on pace deltas clipped to +-2 s/lap (T-053)
PLACEHOLDER_INTERACTION = {
    "version": 0,
    "dirty_air": {"window_s": 1.0, "time_loss_s": 0.3, "extra_deg_frac": 0.15},
    "overtake": {
        "model": "logistic",
        "window_s": 1.5,
        "intercept": -1.5,
        "pace_delta_coef": 3.0,
        "tyre_age_delta_coef": 0.05,
        "gap_coef": -1.0,
        "track_offset": {},
    },
    "pass_time_cost_s": {"attacker": 0.2, "defender": 0.3},
    "lap1": {
        "by_grid_bin": {
            "1-3": {"mean": 0.5, "sd": 1.8},
            "4-6": {"mean": 0.4, "sd": 3.4},
            "7-10": {"mean": 0.6, "sd": 3.4},
            "11-15": {"mean": -0.8, "sd": 2.6},
            "16-22": {"mean": -1.1, "sd": 2.5},
        }
    },
    "min_follow_gap_s": 0.6,
}

# FALLBACK values for things not in the T-053 schema (the T-034 PLACEHOLDERs). T-060 fits them from real data into
# data/sim_params/grid_sim_calibration.json (calibrate_grid_sim.py); load_grid_calibration() returns the fitted value and
# falls back to these per key (GridCalibration.placeholder_keys lists which). Same fallback pattern as the interaction file.
SC_GAP_S = 0.9  # PLACEHOLDER fallback: spacing between cars once bunched behind the SC
SC_GAP_CLOSURE = 0.5  # PLACEHOLDER fallback: share of the extra gap removed per SC lap
LAP1_SPACING_S = 0.7  # PLACEHOLDER fallback: spacing between consecutive cars at the end of lap 1
# PLACEHOLDER fallback race-day pace uncertainty on top of the quali-based pace estimate (team-wide + driver-only), s/lap
TEAM_PACE_SD_S = 0.20
DRIVER_PACE_SD_S = 0.15
# T-060 overtake calibration, identity = the T-053 logistic as fitted (pace delta scale 1, no intercept shift)
OV_PACE_SCALE = 1.0
OV_INTERCEPT_SHIFT = 0.0
# Fallbacks for grid_inputs' pace / strategy-mix constants (see grid_inputs.py for what they mean)
RACE_PACE_SCALE = 0.6
DRIVER_SHRINK = 0.5
QUALI_WEIGHT = 0.5
PLAN_WEIGHTS = (0.6, 0.25, 0.15)
PIT_JITTER_LAPS = 2


@dataclass
class GridCalibration:
    """Calibration constants of the full-grid sim (T-060). Defaults = the T-034 placeholders."""

    team_pace_sd_s: float = TEAM_PACE_SD_S
    driver_pace_sd_s: float = DRIVER_PACE_SD_S
    sc_gap_s: float = SC_GAP_S
    sc_gap_closure: float = SC_GAP_CLOSURE
    lap1_spacing_s: float = LAP1_SPACING_S
    ov_pace_scale: float = OV_PACE_SCALE  # multiplies the pace delta inside the overtake logit
    ov_intercept_shift: float = OV_INTERCEPT_SHIFT  # added to the overtake intercept
    # Gap a car is held at behind the car in front after a FAILED pass attempt (s). None = interaction.min_follow_gap_s,
    # the T-034 behaviour. T-060: min_follow_gap_s is the 5th percentile of follower gaps (0.31 s) but a car that
    # failed to pass sits at a median 0.87 s (sd 0.53) the next lap, so holding it at 0.31 s kept trains inside the
    # attempt window and over-passed; the fitted value is the real median. A single value reset the gap every lap and
    # stopped a clearly faster car closing in, so the held gap is drawn: lognormal(median held_gap_s, sigma
    # held_gap_sigma), from the unused part of the pass-attempt uniform (no extra random numbers, CRN shapes unchanged).
    held_gap_s: Optional[float] = None
    held_gap_sigma: float = 0.0
    race_pace_scale: float = RACE_PACE_SCALE  # grid_inputs
    driver_shrink: float = DRIVER_SHRINK
    quali_weight: float = QUALI_WEIGHT
    plan_weights: tuple = PLAN_WEIGHTS
    pit_jitter_laps: int = PIT_JITTER_LAPS
    source: str = "placeholder"
    placeholder_keys: List[str] = field(default_factory=list)


def load_grid_calibration(path: Optional[Path] = None) -> GridCalibration:
    """
    Fitted calibration (data/sim_params/grid_sim_calibration.json) with a per-key fallback to the module constants
    above. Never raises for a missing or malformed file: it falls back entirely and says so in `source`.
    JSON layout: {"version", "pace": {race_pace_scale, driver_shrink, quali_weight, team_pace_sd_s, driver_pace_sd_s},
    "sc": {gap_s, closure}, "lap1_spacing_s", "strategy_mix": {plan_weights, pit_jitter_laps},
    "overtake_calibration": {pace_delta_scale, intercept_shift}}.
    """
    path = Path(path) if path is not None else CALIBRATION_PATH
    cal = GridCalibration()
    d: dict = {}
    if path.exists():
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            cal.source = f"{path.name} v{d.get('version')}"
        except (OSError, ValueError):
            d, cal.source = {}, "placeholder (unreadable file)"
    else:
        cal.source = "placeholder (no file)"
    missing: List[str] = []

    def get(section, key, attr, cast=float):
        sec = d.get(section) if section else d
        v = sec.get(key) if isinstance(sec, dict) else None
        if v is None or (isinstance(v, float) and v != v):
            missing.append(f"{section}.{key}" if section else key)
            return
        setattr(cal, attr, cast(v))

    get("pace", "race_pace_scale", "race_pace_scale")
    get("pace", "driver_shrink", "driver_shrink")
    get("pace", "quali_weight", "quali_weight")
    get("pace", "team_pace_sd_s", "team_pace_sd_s")
    get("pace", "driver_pace_sd_s", "driver_pace_sd_s")
    get("sc", "gap_s", "sc_gap_s")
    get("sc", "closure", "sc_gap_closure")
    get(None, "lap1_spacing_s", "lap1_spacing_s")
    get("strategy_mix", "plan_weights", "plan_weights", lambda v: tuple(float(x) for x in v))
    get("strategy_mix", "pit_jitter_laps", "pit_jitter_laps", int)
    get("overtake_calibration", "pace_delta_scale", "ov_pace_scale")
    get("overtake_calibration", "intercept_shift", "ov_intercept_shift")
    get("overtake_calibration", "held_gap_s", "held_gap_s")
    get("overtake_calibration", "held_gap_sigma", "held_gap_sigma")
    cal.placeholder_keys = sorted(set(missing)) if d else ["all"]
    return cal


@dataclass
class InteractionParams:
    dirty_window_s: float
    dirty_time_loss_s: float
    dirty_extra_deg_frac: float
    ov_intercept: float
    ov_pace_coef: float
    ov_tyre_age_coef: float
    ov_gap_coef: float
    ov_track_offset: Dict[str, float]
    pass_cost_attacker_s: float
    pass_cost_defender_s: float
    lap1_bins: Dict[str, Dict[str, float]]
    min_follow_gap_s: float
    ov_window_s: float = 1.5
    source: str = "placeholder"
    placeholder_keys: List[str] = field(default_factory=list)

    def lap1_mean_sd(self, grid: np.ndarray):
        """Per-car lap-1 (mean positions gained, sd) arrays for 1-based grid slots."""
        mean = np.zeros(len(grid))
        sd = np.zeros(len(grid))
        for name, lo, hi in LAP1_BINS:
            m = (grid >= lo) & (grid <= hi)
            mean[m] = self.lap1_bins[name]["mean"]
            sd[m] = self.lap1_bins[name]["sd"]
        return mean, sd


def load_interaction_params(path: Optional[Path] = None) -> InteractionParams:
    """
    Fitted interaction parameters (T-053) with PLACEHOLDER fallback per key. Never raises for a missing or
    malformed file: it falls back entirely to placeholders and says so in `source`.
    """
    path = Path(path) if path is not None else INTERACTION_PARAMS_PATH
    ph = PLACEHOLDER_INTERACTION
    d: dict = {}
    source = "placeholder (no file)"
    if path.exists():
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            source = f"{path.name} v{d.get('version')}"
        except (OSError, ValueError):
            d, source = {}, "placeholder (unreadable file)"
    missing: List[str] = []

    def get(*keys):
        fall = ph
        for k in keys:
            fall = fall[k]
        cur = d
        for k in keys:
            if isinstance(cur, dict) and cur.get(k) is not None:
                cur = cur[k]
            else:
                missing.append(".".join(keys))
                return fall
        return cur

    ov_model = d.get("overtake", {}).get("model", "logistic")
    if ov_model != "logistic":
        raise ValueError(f"unsupported overtake model {ov_model!r} in {path}")
    bins = {}
    for name, _, _ in LAP1_BINS:
        b = d.get("lap1", {}).get("by_grid_bin", {}).get(name)
        if b and b.get("mean") is not None and b.get("sd") is not None:
            bins[name] = {"mean": float(b["mean"]), "sd": float(b["sd"])}
        else:
            missing.append(f"lap1.by_grid_bin.{name}")
            bins[name] = dict(ph["lap1"]["by_grid_bin"][name])
    return InteractionParams(
        dirty_window_s=float(get("dirty_air", "window_s")),
        dirty_time_loss_s=float(get("dirty_air", "time_loss_s")),
        dirty_extra_deg_frac=float(get("dirty_air", "extra_deg_frac")),
        ov_intercept=float(get("overtake", "intercept")),
        ov_pace_coef=float(get("overtake", "pace_delta_coef")),
        ov_tyre_age_coef=float(get("overtake", "tyre_age_delta_coef")),
        ov_gap_coef=float(get("overtake", "gap_coef")),
        ov_track_offset={k: float(v) for k, v in (d.get("overtake", {}).get("track_offset") or {}).items()},
        pass_cost_attacker_s=float(get("pass_time_cost_s", "attacker")),
        pass_cost_defender_s=float(get("pass_time_cost_s", "defender")),
        lap1_bins=bins,
        min_follow_gap_s=float(get("min_follow_gap_s")),
        ov_window_s=float(get("overtake", "window_s")),
        source=source,
        placeholder_keys=sorted(set(missing)) if path.exists() and d else ["all"],
    )


@dataclass
class CarSpec:
    """One car on the grid. pace_delta_s: seconds per lap slower than the reference (>= 0 typically)."""

    code: str
    team: str
    grid: int  # 1-based starting slot; pit-lane starters should be given the last slot
    pace_delta_s: float
    strategy: List[Stint]
    p_dnf: float = 0.0


@dataclass
class GridSimResult:
    codes: List[str]
    teams: List[str]
    grid: np.ndarray  # (D,)
    positions: np.ndarray  # (N, D) int16 finishing position 1..D (retirements last)
    finish_time_s: np.ndarray  # (N, D) float, NaN for retirements
    retired: np.ndarray  # (N, D) bool
    n_overtakes: np.ndarray  # (N,) successful passes per run (diagnostic)
    n_sc_laps: np.ndarray  # (N,) SC laps per run
    total_laps: int
    interaction: InteractionParams

    @property
    def n_sims(self) -> int:
        return self.positions.shape[0]

    def position_distribution(self) -> np.ndarray:
        """(D cars, D positions) probabilities."""
        d = self.positions.shape[1]
        out = np.zeros((d, d))
        for p in range(1, d + 1):
            out[:, p - 1] = (self.positions == p).mean(axis=0)
        return out

    def summary(self) -> List[dict]:
        pos = self.positions
        pts = FIA_POINTS[np.minimum(pos, len(FIA_POINTS)) - 1]
        dist = self.position_distribution()
        rows = []
        for j, code in enumerate(self.codes):
            fin = self.finish_time_s[:, j]
            ok = ~np.isnan(fin)
            p10, p50, p90 = (np.percentile(fin[ok], [10, 50, 90]) if ok.any() else (np.nan,) * 3)
            rows.append(
                {
                    "code": code,
                    "team": self.teams[j],
                    "grid": int(self.grid[j]),
                    "p_win": float((pos[:, j] == 1).mean()),
                    "p_podium": float((pos[:, j] <= 3).mean()),
                    "p_top10": float((pos[:, j] <= 10).mean()),
                    "p_dnf": float(self.retired[:, j].mean()),
                    "exp_points": float(pts[:, j].mean()),
                    "mean_position": float(pos[:, j].mean()),
                    "p10_time_s": float(p10),
                    "p50_time_s": float(p50),
                    "p90_time_s": float(p90),
                    "position_probs": [float(x) for x in dist[j]],
                }
            )
        return rows


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _strategy_tables(strategies: Sequence[List[Stint]], tire_model: dict, total_laps: int):
    """
    Per car, per lap deterministic tables (D, L): tyre delta incl. compound offset (s), tyre age, and the
    out-lap flag (a stop was made at the end of the previous lap).
    """
    d = len(strategies)
    tyre = np.zeros((d, total_laps))
    age = np.zeros((d, total_laps), dtype=np.int32)
    pit = np.zeros((d, total_laps), dtype=bool)
    for j, strat in enumerate(strategies):
        n = sum(s.laps for s in strat)
        if n != total_laps:
            raise ValueError(f"car {j}: strategy covers {n} laps, race is {total_laps}")
        lap = 0
        for k, s in enumerate(strat):
            off = compound_pace_offset(tire_model, s.compound)
            for i in range(s.laps):
                a = s.start_age + i
                tyre[j, lap] = predict_delta(tire_model, s.compound, a) + off
                age[j, lap] = a
                pit[j, lap] = i == 0 and k > 0
                lap += 1
    return tyre, age, pit


def simulate_grid(
    cars: Sequence[CarSpec],
    base_laptime_s: float,
    tire_model: dict,
    total_laps: int,
    n_sims: int = 2000,
    seed: Optional[int] = 0,
    pit_loss_s: float = DEFAULT_PIT_LOSS_SECONDS,
    pit_loss_noise_std: float = DEFAULT_PIT_LOSS_NOISE_STD,
    lap_time_noise_std: float = DEFAULT_LAP_TIME_NOISE_STD,
    interaction: Optional[InteractionParams] = None,
    safety_cars: Optional[SafetyCarParams] = None,
    forced_sc_lap: Optional[int] = None,
    forced_sc_laps: Optional[int] = None,
    track_id: Optional[str] = None,
    pace_noise: bool = True,
    dnfs: bool = True,
    interactions: bool = True,
    plan_hook=None,
    calibration: Optional[GridCalibration] = None,
) -> GridSimResult:
    """
    Simulate the whole grid n_sims times. See the module docstring for the model.

    cars: one CarSpec per car; grid slots must be distinct. Strategy of every car must sum to total_laps.
    base_laptime_s: same convention as race_simulator (HARD, fresh, empty tank, clean air) for a car whose
        pace_delta_s is 0.
    safety_cars: fitted SC/VSC parameters; None = no random neutralisations. forced_sc_lap (1-indexed lap
        the SC starts on) forces an SC for forced_sc_laps laps (default: median fitted SC length) on top
        of any random ones; without `safety_cars` the forced SC is the only one.
    interactions=False turns off dirty air, overtakes and the lap-1 shuffle (cars then simply run their
        own clock; used by tests and as a no-interaction reference).
    plan_hook (T-055, optional): callable(sc, vsc, (tyre, age, pit)) -> {car index: (tyre, age, pit)} with
        per-run (N, L) tables for those cars, called once after the SC/VSC timeline is drawn. It lets a
        reactive pit policy (pit_policy.py) change a car's stops per run depending on the timeline; cars not
        in the returned dict keep their fixed plan. None = fixed plans, bit-identical to before.
    calibration (T-060): pace noise, SC bunching, lap-1 spacing and the overtake pace-scale / intercept-shift;
        default load_grid_calibration() = data/sim_params/grid_sim_calibration.json with placeholder fallbacks.
    """
    ip = interaction if interaction is not None else load_interaction_params()
    cal = calibration if calibration is not None else load_grid_calibration()
    n, d, nl = int(n_sims), len(cars), int(total_laps)
    if d < 2:
        raise ValueError("need at least 2 cars")
    grid = np.array([c.grid for c in cars], dtype=int)
    if len(set(grid.tolist())) != d:
        raise ValueError("grid slots must be distinct")
    rng = np.random.default_rng(seed)
    rows = np.arange(n)[:, None]

    tyre, age, pit = _strategy_tables(
        [c.strategy for c in cars], tire_model, nl
    )  # (D, L)
    fuel = np.array([fuel_effect(tire_model, lap, nl) for lap in range(1, nl + 1)])
    pace = np.array([c.pace_delta_s for c in cars])
    teams = [c.team for c in cars]
    team_idx = np.array([sorted(set(teams)).index(t) for t in teams])

    # ---- draws that do not depend on strategies / SC / interaction settings (common random numbers)
    z_team = rng.standard_normal((n, len(set(teams))))
    z_drv = rng.standard_normal((n, d))
    z_lap1 = rng.standard_normal((n, d))
    u_dnf = rng.random((n, d))
    u_dnf_lap = rng.random((n, d))
    sc_seed = int(rng.integers(2**63))

    pace_run = np.broadcast_to(pace, (n, d)).copy()
    if pace_noise:
        pace_run += cal.team_pace_sd_s * z_team[:, team_idx] + cal.driver_pace_sd_s * z_drv

    p_dnf = np.array([c.p_dnf for c in cars])
    if dnfs:
        dnf_lap = np.where(u_dnf < p_dnf, np.floor(u_dnf_lap * nl).astype(int), nl + 1)  # 0-indexed lap of retirement
    else:
        dnf_lap = np.full((n, d), nl + 1)

    sc_params = safety_cars
    sc = np.zeros((n, nl), dtype=bool)
    vsc = np.zeros((n, nl), dtype=bool)
    if safety_cars is not None:
        sc, vsc = draw_timelines(np.random.default_rng(sc_seed), n, nl, safety_cars)
    if forced_sc_lap is not None:
        if sc_params is None:
            from app.core.safety_cars import load_params

            sc_params = load_params()
        sc, vsc = apply_forced_sc(
            sc, vsc, forced_sc_lap, forced_sc_laps if forced_sc_laps is not None else default_sc_length(sc_params)
        )
    if sc_params is None:
        from app.core.safety_cars import load_params

        sc_params = load_params()  # only its multiples/ratios are used, and only if sc/vsc masks are non-empty

    overrides = plan_hook(sc, vsc, (tyre, age, pit)) if plan_hook is not None else {}

    # ---- lap 1: grid order perturbed by the fitted position change
    mean1, sd1 = ip.lap1_mean_sd(grid)
    if interactions:
        score = grid[None, :] + (mean1[None, :] + sd1[None, :] * z_lap1)  # new position = grid + N(mean, sd), then re-rank
        # ties (equal score) broken by grid slot; argsort of argsort = rank
        key = score + 1e-9 * grid[None, :]
        rank1 = np.argsort(np.argsort(key, axis=1), axis=1)  # 0-based
    else:
        rank1 = np.broadcast_to(np.argsort(np.argsort(grid)), (n, d))
    lap1_base = base_laptime_s + tyre[:, 0] + fuel[0]  # (D,)
    t_lead = float(lap1_base.min())
    T = t_lead + rank1 * cal.lap1_spacing_s
    T = T + np.where(dnf_lap <= 0, DNF_BIG_S, 0.0)

    mf = ip.min_follow_gap_s
    held_gap = cal.held_gap_s if cal.held_gap_s is not None else mf
    held_gap_sigma = cal.held_gap_sigma if cal.held_gap_s is not None else 0.0
    from scipy.special import ndtri as _ndtri  # inverse normal cdf, only needed for the held-gap draw
    window = ip.dirty_window_s
    offset = ip.ov_track_offset.get(track_id, 0.0) if track_id else 0.0
    n_pass = np.zeros(n, dtype=np.int32)
    n_sc_laps = sc.sum(axis=1).astype(np.int32)

    for lap in range(1, nl):
        sc_l, vsc_l = sc[:, lap], vsc[:, lap]
        neutral = sc_l | vsc_l
        noise = rng.standard_normal((n, d)) if lap_time_noise_std else np.zeros((n, d))
        pit_z = rng.standard_normal((n, d)) if pit_loss_noise_std else np.zeros((n, d))
        u_ov = rng.random((n, d))
        dead = dnf_lap <= lap  # (N, D) car-order

        order = np.argsort(T, axis=1, kind="stable")  # position -> car
        Ts = np.take_along_axis(T, order, axis=1)
        gap = np.empty_like(Ts)
        gap[:, 0] = np.inf
        gap[:, 1:] = Ts[:, 1:] - Ts[:, :-1]

        def g(a):  # car-order (N, D) -> position order
            return np.take_along_axis(a, order, axis=1)

        if overrides:  # reactive cars: per-run tables for this lap (T-055)
            tyre_c = np.broadcast_to(tyre[:, lap], (n, d)).copy()
            age_c = np.broadcast_to(age[:, lap], (n, d)).copy()
            pit_c = np.broadcast_to(pit[:, lap], (n, d)).copy()
            for j, (ty_j, ag_j, pi_j) in overrides.items():
                tyre_c[:, j], age_c[:, j], pit_c[:, j] = ty_j[:, lap], ag_j[:, lap], pi_j[:, lap]
            tyre_l, age_l, pit_l = g(tyre_c), g(age_c), g(pit_c)
        else:
            tyre_l = tyre[:, lap][order]  # (N, D)
            age_l = age[:, lap][order]
            pit_l = pit[:, lap][order]
        pace_s = g(pace_run)
        dead_s = g(dead)
        noise_s = g(noise)
        pitz_s = g(pit_z)
        uov_s = g(u_ov)

        green = base_laptime_s + fuel[lap] + tyre_l + pace_s + noise_s  # free lap, clean air
        if interactions:
            dirty = (gap < window) & ~neutral[:, None]
            lap_t = green + dirty * (ip.dirty_time_loss_s + ip.dirty_extra_deg_frac * np.maximum(tyre_l, 0.0))
        else:
            lap_t = green.copy()
        # neutralised laps: nobody races; all cars run the same neutral lap (no pace delta, no noise)
        neutral_base = base_laptime_s + fuel[lap] + tyre_l
        ratio = np.where(sc_l, sc_params.sc_pit_loss_ratio, np.where(vsc_l, sc_params.vsc_pit_loss_ratio, 1.0))
        mult = np.where(sc_l, sc_params.sc_lap_multiple, np.where(vsc_l, sc_params.vsc_lap_multiple, 1.0))
        pit_cost = np.where(
            pit_l, (pit_loss_s + pit_loss_noise_std * pitz_s).clip(min=0.0) * ratio[:, None], 0.0
        )
        lap_t = np.where(neutral[:, None], neutral_base * mult[:, None], lap_t) + pit_cost
        F = np.where(dead_s, DNF_BIG_S, lap_t)
        Fend = Ts + F

        R = Fend.copy()
        if interactions:
            # --- green racing: resolve catches and overtakes front to back
            m1 = np.full(n, -np.inf)  # time of the last car in the line so far
            m2 = np.full(n, -np.inf)  # time of the car before it
            tail = np.zeros(n, dtype=int)  # position index of the last car in the line
            racing = ~neutral
            ar = np.arange(n)
            for p in range(d):
                fe = Fend[:, p]
                # a car that is stopping this lap leaves the line (cars behind pass it for free) and
                # is not held by anyone: its pit loss already puts it behind
                alive = ~dead_s[:, p] & ~pit_l[:, p]
                catch = alive & racing & np.isfinite(m1) & ((Ts[:, p] - Ts[ar, tail]) <= ip.ov_window_s)
                cidx = np.nonzero(catch)[0]
                free = alive & ~catch
                if cidx.size:
                    t_c = tail[cidx]
                    # defender pace - attacker pace (s/lap), tyre/fuel-normalised, clipped as in the T-053 fit
                    pdelta = np.clip(pace_s[cidx, t_c] - pace_s[cidx, p], -PACE_DELTA_CLIP_S, PACE_DELTA_CLIP_S)
                    adelta = age_l[cidx, t_c] - age_l[cidx, p]
                    gp = np.minimum(Ts[cidx, p] - Ts[cidx, t_c], 30.0)  # start-of-lap gap to the car in front (the line's tail)
                    logit = (
                        ip.ov_intercept + cal.ov_intercept_shift + ip.ov_pace_coef * cal.ov_pace_scale * pdelta
                        + ip.ov_tyre_age_coef * adelta + ip.ov_gap_coef * gp + offset
                    )
                    win = uov_s[cidx, p] < _sigmoid(logit)
                    m1c, m2c = m1[cidx], m2[cidx]
                    # success: attacker ends the lap ahead of the defender (never slower than its own free lap
                    # allows, never through the car before the defender); failure: held at min_follow_gap
                    r_succ = np.maximum(np.minimum(fe[cidx] + ip.pass_cost_attacker_s, m1c - 0.5 * mf), m2c + mf)
                    if held_gap_sigma > 0.0:
                        # given a failed attempt (u >= p), (u - p) / (1 - p) is U(0, 1): the quantile of the held-gap law
                        pc = _sigmoid(logit)
                        u_f = np.clip((uov_s[cidx, p] - pc) / (1.0 - pc), 1e-6, 1.0 - 1e-6)
                        hg = np.maximum(mf, held_gap * np.exp(held_gap_sigma * _ndtri(u_f)))
                    else:
                        hg = held_gap
                    r_fail = np.maximum(fe[cidx], m1c + hg)
                    R[cidx, p] = np.where(win, r_succ, r_fail)
                    wi = cidx[win]
                    if wi.size:
                        tw = tail[wi]
                        R[wi, tw] = np.maximum(R[wi, tw] + ip.pass_cost_defender_s, r_succ[win] + mf)
                        n_pass[wi] += 1
                    # tail bookkeeping: success -> defender (new tail value) stays tail, attacker becomes 2nd
                    new_m1 = np.where(win, R[cidx, tail[cidx]], r_fail)
                    new_m2 = np.where(win, r_succ, m1c)
                    m1[cidx], m2[cidx] = new_m1, new_m2
                    tail[cidx[~win]] = p
                fi = np.nonzero(free)[0]
                if fi.size:
                    m2[fi] = m1[fi]
                    m1[fi] = fe[fi]
                    tail[fi] = p
            # --- safety car: close up behind the leader; VSC: freeze (already Fend, no catches)
            scr = np.nonzero(sc_l)[0]
            if scr.size:
                Fs, Tss, deads = F[scr], Ts[scr], dead_s[scr]
                out = Fend[scr].copy()
                prev_end = out[:, 0].copy()
                prev_F = Fs[:, 0].copy()
                prev_start = Tss[:, 0].copy()
                for p in range(1, d):
                    gap_start = Tss[:, p] - prev_start
                    new_gap = np.maximum(cal.sc_gap_s, gap_start * (1.0 - cal.sc_gap_closure) + (Fs[:, p] - prev_F))
                    cur = np.where(deads[:, p], Fend[scr][:, p], prev_end + new_gap)
                    out[:, p] = cur
                    ok = ~deads[:, p]
                    prev_end = np.where(ok, cur, prev_end)
                    prev_F = np.where(ok, Fs[:, p], prev_F)
                    prev_start = np.where(ok, Tss[:, p], prev_start)
                R[scr] = out
            # --- VSC: keep order (a car cannot pass), keep gaps; Fend already has them
        # write back to car order
        Tn = np.empty_like(T)
        np.put_along_axis(Tn, order, R, axis=1)
        T = Tn

    order = np.argsort(T, axis=1, kind="stable")
    positions = np.empty((n, d), dtype=np.int16)
    np.put_along_axis(positions, order, np.broadcast_to(np.arange(1, d + 1, dtype=np.int16), (n, d)).copy(), axis=1)
    retired = dnf_lap < nl
    finish = np.where(retired, np.nan, T)
    return GridSimResult(
        codes=[c.code for c in cars],
        teams=teams,
        grid=grid,
        positions=positions,
        finish_time_s=finish,
        retired=retired,
        n_overtakes=n_pass,
        n_sc_laps=n_sc_laps,
        total_laps=nl,
        interaction=ip,
    )


def rank_by_points(exp_points, mean_position) -> np.ndarray:
    """1-based predicted order: expected FIA points desc, mean position as the tie-break (same rule as
    predictor.rank_by_points, T-045)."""
    exp_points, mean_position = np.asarray(exp_points, float), np.asarray(mean_position, float)
    order = np.lexsort((mean_position, -exp_points))
    rank = np.empty(len(order), dtype=int)
    rank[order] = np.arange(1, len(order) + 1)
    return rank
