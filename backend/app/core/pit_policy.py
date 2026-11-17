"""
Safety-car-reactive pit policy (T-055, Roadmap Phase 2/4).

Every other part of the engine plays a fixed stint plan. Real teams do not: 44% of 2026 top-10 stops were
made under an SC/VSC (T-049), because a stop under neutralisation costs far less time (fitted
pit_loss_ratio). This module adds one reactive layer on top of a base plan:

    "Run the base plan. If an SC (or VSC) STARTS on a lap in [a, b] and my tyres are at least k laps old,
     box now for compound C. Otherwise keep the plan."

What "box now" does to the plan (pull-forward): the stop the plan would have made next is pulled forward to
the SC lap, and the new stint on C runs to where the plan's following stint ended (so the number of stops
is unchanged). If the car is on the last stint of the plan there is no stop to pull forward, so it is an
extra stop and the new stint runs to the flag. A reaction is skipped (the policy "does not fire") when it
would leave a single compound in the plan (two-compound rule), would put a stint on C longer than that
compound's cap at the track (never extrapolate past observed stints), or the SC lap is the plan's own stop
lap / lap 1. The new tyre goes on at the start of the SC lap (the stop's out-lap carries the SC-discounted
pit loss, the grid sim's convention); that is one lap more optimistic than boxing at the end of it.

Search: dynamic programming over (lap, tyre age, compound, stops made) would give the optimal action for
every state, but the grid sim is the expensive part (a state's value needs a full-field race), and the
requested policy class has four numbers. So the search is a coarse grid, made cheap by one fact: the outcome
of a run under policy (a, b, k, C) is exactly the outcome of "fire at event e", where e is the first
eligible SC start in the window. So for each (k, C) the grid sim is run once per event SLOT ("fire at the
j-th eligible SC/VSC start of the run", j = 0, 1, 2), with common random numbers, which records the per-run,
per-event gain for ALL windows at once; the window [a, b] and the SC-only/any trigger are then optimised on
those tables with no further simulation. Selection uses one seed, the reported gain comes from a fresh
seed (paired against the same base plan, common random numbers), so the interval is not inflated by the
choice among ~10^3 windows.

Scored with the full-grid sim (race_grid_sim.simulate_grid) through its additive `plan_hook`, so track
position, bunching behind the SC, and the field's own (fixed-plan) SC stops all matter. `field_reacts`
re-scores with every other car running the same rule (nobody gains from a free stop that everybody takes).

Simplifications / PLACEHOLDERS (labelled): at most MAX_SLOTS events per run are considered; the search grids
(K_GRID, window fractions) are coarse by design; tyre wear and fuel are not reduced under SC (engine-wide);
one policy per car, no reaction to the other cars' positions.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.core.race_grid_sim import FIA_POINTS, CarSpec, GridSimResult, _strategy_tables, simulate_grid
from app.core.race_simulator import Stint

COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
PLURAL = {"SOFT": "Softs", "MEDIUM": "Mediums", "HARD": "Hards"}
MIN_CAP_LAPS = 15  # floor on a compound's stint cap, same as optimizer.MIN_STINT_CAP
MIN_LAPS_AFTER = 5  # PLACEHOLDER: a reaction that leaves fewer laps than this to run on the new set is skipped
MAX_SLOTS = 3  # SC/VSC starts per run considered by the search (a run with a 4th event is vanishingly rare)
K_GRID = (4, 8, 12, 16)  # PLACEHOLDER: tyre-age thresholds searched (laps)
LO_FRACS = tuple(np.round(np.arange(0.05, 0.80, 0.05), 2))
HI_FRACS = tuple(np.round(np.arange(0.15, 0.95, 0.05), 2))
Z95 = 1.96


@dataclass(frozen=True)
class PitPolicy:
    """Box for `compound` when an SC (or VSC) starts on a lap in [window_lo, window_hi] and tyre age >= min_age."""

    window_lo: int
    window_hi: int
    min_age: int
    compound: str
    trigger: str = "sc_or_vsc"  # "sc" (full safety car only) or "sc_or_vsc"

    def __post_init__(self):
        if self.compound not in COMPOUNDS:
            raise ValueError(f"compound must be one of {COMPOUNDS}")
        if self.trigger not in ("sc", "sc_or_vsc"):
            raise ValueError("trigger must be 'sc' or 'sc_or_vsc'")
        if self.window_lo < 2 or self.window_hi < self.window_lo:
            raise ValueError("window must satisfy 2 <= window_lo <= window_hi")
        if self.min_age < 0:
            raise ValueError("min_age must be >= 0")

    def describe(self, plan: Optional[Sequence[Stint]] = None) -> str:
        ev = "a safety car" if self.trigger == "sc" else "a safety car or virtual safety car"
        rule = (
            f"If {ev} comes out between laps {self.window_lo} and {self.window_hi} and your tyres are at least "
            f"{self.min_age} laps old, box for {PLURAL[self.compound]}."
        )
        if plan is not None:
            rule += " Otherwise keep the plan (" + "-".join(f"{s.compound[0]}{s.laps}" for s in plan) + ")."
        return rule


def stint_caps(track: dict, total_laps: int) -> Dict[str, int]:
    """Per-compound stint cap used by the optimizer (p90 of observed stints, floored at 15, clipped to the race)."""
    raw = dict(track.get("max_stint_laps") or {})
    for c, v in (track.get("typical_stint_laps") or {}).items():
        if v is not None and c in raw:
            raw[c] = min(int(raw[c]), int(v))
    return {c: min(max(int(v), MIN_CAP_LAPS), total_laps) for c, v in raw.items() if c in COMPOUNDS}


def tyre_age_on_lap(plan: Sequence[Stint], lap: int) -> Optional[int]:
    """Age of the tyre on 1-indexed `lap`, or None when `lap` starts a stint (a planned stop) or is out of range."""
    start = 1
    for s in plan:
        end = start + s.laps - 1
        if start <= lap <= end:
            return None if (lap == start and start > 1) else s.start_age + (lap - start)
        start = end + 1
    return None


def apply_policy_to_plan(
    plan: Sequence[Stint], lap: int, compound: str, caps: Optional[Dict[str, int]] = None
) -> Optional[List[Stint]]:
    """
    The plan after "box on `lap` for `compound`" (see the module docstring), or None when the reaction is not
    allowed (planned stop lap, one compound left, new stint longer than the compound's cap, too few laps left).
    """
    total = sum(s.laps for s in plan)
    start = 1
    for i, s in enumerate(plan):
        end = start + s.laps - 1
        if start <= lap <= end:
            break
        start = end + 1
    else:
        return None
    if lap == start:  # the plan already stops on this lap (or lap 1): nothing to react with
        return None
    last = i == len(plan) - 1
    stop_end = total if last else start + s.laps - 1 + plan[i + 1].laps
    if total - lap + 1 < MIN_LAPS_AFTER:
        return None
    new_len = stop_end - lap + 1
    if caps and compound in caps and new_len > caps[compound]:
        return None
    out = [Stint(p.compound, p.laps, p.start_age) for p in plan[:i]]
    out.append(Stint(s.compound, lap - start, s.start_age))
    out.append(Stint(compound, new_len, 1))
    out.extend(Stint(p.compound, p.laps, p.start_age) for p in plan[i + 2 :])
    if len({x.compound for x in out}) < 2:
        return None
    return out


class PolicyHook:
    """
    The `plan_hook` for simulate_grid: turns the drawn SC/VSC timeline into per-run (N, L) tables for the
    cars that follow `policy` (car index -> base plan in `plans`). After the sim, `fire_lap[j]` is the
    1-indexed lap each run's policy fired on (0 = never) and `fire_is_sc[j]` whether that event was a full SC.

    slot: search mode. Ignores the window and trigger and fires at the slot-th eligible SC/VSC start of each
    run (eligible = age >= min_age and the reaction is allowed), which records per-event outcomes.
    """

    def __init__(
        self,
        policy: PitPolicy,
        plans: Dict[int, Sequence[Stint]],
        tire_model: dict,
        total_laps: int,
        caps: Optional[Dict[str, int]] = None,
        slot: Optional[int] = None,
    ):
        self.policy, self.plans, self.tire_model = policy, dict(plans), tire_model
        self.nl, self.caps, self.slot = int(total_laps), caps, slot
        self.fire_lap: Dict[int, np.ndarray] = {}
        self.fire_is_sc: Dict[int, np.ndarray] = {}

    def _car_tables(self, plan: Sequence[Stint]):
        """Eligibility per 0-indexed lap i (fire on lap i+1) and the stacked modified tables, row nl = base plan."""
        nl, p = self.nl, self.policy
        elig = np.zeros(nl, dtype=bool)
        ty = np.zeros((nl + 1, nl))
        ag = np.zeros((nl + 1, nl), dtype=np.int32)
        pi = np.zeros((nl + 1, nl), dtype=bool)
        bt, ba, bp = _strategy_tables([list(plan)], self.tire_model, nl)
        ty[nl], ag[nl], pi[nl] = bt[0], ba[0], bp[0]
        ty[:nl], ag[:nl], pi[:nl] = bt[0], ba[0], bp[0]
        for lap in range(2, nl):
            age = tyre_age_on_lap(plan, lap)
            if age is None or age < p.min_age:
                continue
            new = apply_policy_to_plan(plan, lap, p.compound, self.caps)
            if new is None:
                continue
            t, a, q = _strategy_tables([new], self.tire_model, nl)
            ty[lap - 1], ag[lap - 1], pi[lap - 1] = t[0], a[0], q[0]
            elig[lap - 1] = True
        return elig, ty, ag, pi

    def __call__(self, sc, vsc, base_tables):
        n, nl = sc.shape
        p = self.policy
        start_sc = np.zeros((n, nl), dtype=bool)
        start_vsc = np.zeros((n, nl), dtype=bool)
        start_sc[:, 1:] = sc[:, 1:] & ~sc[:, :-1]
        start_vsc[:, 1:] = vsc[:, 1:] & ~vsc[:, :-1]
        laps = np.arange(1, nl + 1)
        if self.slot is None:
            ev = start_sc if p.trigger == "sc" else (start_sc | start_vsc)
            ev = ev & ((laps >= p.window_lo) & (laps <= p.window_hi))[None, :]
        else:
            ev = start_sc | start_vsc
        out = {}
        for j, plan in self.plans.items():
            elig, ty, ag, pi = self._car_tables(plan)
            m = ev & elig[None, :]
            if self.slot is None:
                has = m.any(axis=1)
                first = m.argmax(axis=1)
            else:
                c = np.cumsum(m, axis=1)
                sel = m & (c == self.slot + 1)
                has = sel.any(axis=1)
                first = sel.argmax(axis=1)
            pick = np.where(has, first, nl)  # row nl = base plan
            self.fire_lap[j] = np.where(has, first + 1, 0)
            self.fire_is_sc[j] = has & start_sc[np.arange(n), first]
            out[j] = (ty[pick], ag[pick], pi[pick])
        return out


def points_of(res: GridSimResult, j: int) -> np.ndarray:
    return FIA_POINTS[np.minimum(res.positions[:, j], len(FIA_POINTS)) - 1]


def mean_ci(x: np.ndarray) -> Tuple[float, float]:
    """Mean and 95% half-width (normal approximation) of paired per-run differences."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float(x.mean()) if len(x) else 0.0, 0.0
    return float(x.mean()), float(Z95 * x.std(ddof=1) / np.sqrt(len(x)))


@dataclass
class PolicyContext:
    """Everything fixed while policies are compared: the field, the hero car and its base plan, the race."""

    cars: List[CarSpec]
    hero: int
    base_laptime_s: float
    tire_model: dict
    total_laps: int
    pit_loss_s: float
    caps: Dict[str, int]
    safety_cars: Optional[object]  # SafetyCarParams or None (= no random SC/VSC)
    interaction: Optional[object] = None
    track_id: Optional[str] = None
    simulate_kwargs: dict = field(default_factory=dict)

    @property
    def hero_plan(self) -> List[Stint]:
        return self.cars[self.hero].strategy

    def run(
        self,
        policy: Optional[PitPolicy],
        n_sims: int,
        seed: int,
        slot: Optional[int] = None,
        field_policy: Optional[PitPolicy] = None,
    ):
        """One grid sim. policy applies to the hero; field_policy (optional) to every other car. Returns (result, hook)."""
        plans = {}
        if policy is not None:
            plans[self.hero] = self.hero_plan
        hook = PolicyHook(policy, plans, self.tire_model, self.total_laps, self.caps, slot) if plans else None
        hooks = [hook] if hook else []
        if field_policy is not None:
            fplans = {j: c.strategy for j, c in enumerate(self.cars) if j != self.hero}
            hooks.append(PolicyHook(field_policy, fplans, self.tire_model, self.total_laps, self.caps))

        def combined(sc, vsc, base):
            merged = {}
            for h in hooks:
                merged.update(h(sc, vsc, base))
            return merged

        res = simulate_grid(
            self.cars, self.base_laptime_s, self.tire_model, self.total_laps, n_sims=n_sims, seed=seed,
            pit_loss_s=self.pit_loss_s, interaction=self.interaction, safety_cars=self.safety_cars,
            track_id=self.track_id, plan_hook=combined if hooks else None, **self.simulate_kwargs,
        )
        return res, hook


# --------------------------------------------------------------------------- search


@dataclass
class EventTable:
    """Per-run outcome of firing at the slot-th eligible event, for one (min_age, compound)."""

    lap: np.ndarray  # (N,) 1-indexed lap, 0 = no such event
    is_sc: np.ndarray
    dpts: np.ndarray
    dpos: np.ndarray  # positions gained (+ = better)


def window_candidates(nl: int) -> Tuple[List[int], List[int]]:
    lo = sorted({2} | {max(2, int(round(f * nl))) for f in LO_FRACS})
    hi = sorted({int(round(f * nl)) for f in HI_FRACS} | {nl - 1})
    return lo, hi


def build_event_tables(ctx: PolicyContext, n_screen: int, seed: int, k_grid=K_GRID, compounds=COMPOUNDS):
    base, _ = ctx.run(None, n_screen, seed)
    pts_b, pos_b = points_of(base, ctx.hero), base.positions[:, ctx.hero].astype(float)
    tables: Dict[Tuple[int, str], List[EventTable]] = {}
    for k in k_grid:
        for c in compounds:
            pol = PitPolicy(2, ctx.total_laps - 1, k, c, "sc_or_vsc")
            slots = []
            for s in range(MAX_SLOTS):
                res, hook = ctx.run(pol, n_screen, seed, slot=s)
                lap = hook.fire_lap[ctx.hero]
                if not lap.any():
                    break
                slots.append(
                    EventTable(lap, hook.fire_is_sc[ctx.hero],
                               points_of(res, ctx.hero) - pts_b, pos_b - res.positions[:, ctx.hero])
                )
            tables[(k, c)] = slots
    return tables


def window_gain(slots: List[EventTable], lo: int, hi: int, trigger: str, field_name: str = "dpts"):
    """Per-run gain of policy (lo, hi, trigger) from the slot tables: the first slot whose event is in the window."""
    n = len(slots[0].lap) if slots else 0
    gain = np.zeros(n)
    taken = np.zeros(n, dtype=bool)
    for s in slots:
        ok = (s.lap >= lo) & (s.lap <= hi) & ~taken
        if trigger == "sc":
            ok &= s.is_sc
        gain[ok] = getattr(s, field_name)[ok]
        taken |= ok
    return gain, taken


def best_policy_from_tables(tables, nl: int, min_fire_rate: float = 0.0):
    """Exhaustive search over (k, C, window, trigger) on the slot tables. Returns (policy, screen gain, screen fire rate)."""
    los, his = window_candidates(nl)
    best = (None, -np.inf, 0.0)
    for (k, c), slots in tables.items():
        if not slots:
            continue
        for trig in ("sc", "sc_or_vsc"):
            for lo in los:
                for hi in his:
                    if hi < lo:
                        continue
                    g, fired = window_gain(slots, lo, hi, trig)
                    m = float(g.mean())
                    # tie-break toward the wider window (a rule that fires more often is simpler and less overfit)
                    if m > best[1] + 1e-12 or (abs(m - best[1]) <= 1e-12 and best[0] is not None and hi - lo > best[0].window_hi - best[0].window_lo):
                        rate = float(fired.mean())
                        if rate >= min_fire_rate:
                            best = (PitPolicy(lo, hi, k, c, trig), m, rate)
    return best


@dataclass
class PolicyEvaluation:
    policy: Optional[PitPolicy]
    n_sims: int
    seed: int
    gain_pts: float
    gain_pts_ci: float
    gain_pos: float
    gain_pos_ci: float
    fire_rate: float
    gain_pts_when_fired: float
    gain_pts_when_fired_ci: float
    gain_pos_when_fired: float
    fire_lap_mean: Optional[float]
    base_exp_points: float
    base_mean_position: float
    p_gain: float  # share of fired runs where the points were strictly higher
    p_loss: float


def evaluate_policy(
    ctx: PolicyContext, policy: PitPolicy, n_sims: int, seed: int, field_reacts: bool = False
) -> PolicyEvaluation:
    """Paired (common random numbers) comparison of policy vs the fixed base plan for the hero car."""
    fp = policy if field_reacts else None
    base, _ = ctx.run(None, n_sims, seed, field_policy=fp)
    res, hook = ctx.run(policy, n_sims, seed, field_policy=fp)
    j = ctx.hero
    dp = points_of(res, j) - points_of(base, j)
    dpos = base.positions[:, j].astype(float) - res.positions[:, j]
    fired = hook.fire_lap[j] > 0
    g, gc = mean_ci(dp)
    gp, gpc = mean_ci(dpos)
    if fired.any():
        gf, gfc = mean_ci(dp[fired])
        gpf = float(dpos[fired].mean())
        lap_mean = float(hook.fire_lap[j][fired].mean())
        p_gain, p_loss = float((dp[fired] > 0).mean()), float((dp[fired] < 0).mean())
    else:
        gf = gfc = gpf = 0.0
        lap_mean, p_gain, p_loss = None, 0.0, 0.0
    return PolicyEvaluation(
        policy, n_sims, seed, g, gc, gp, gpc, float(fired.mean()), gf, gfc, gpf, lap_mean,
        float(points_of(base, j).mean()), float(base.positions[:, j].mean()), p_gain, p_loss,
    )


@dataclass
class PolicySearchResult:
    best: Optional[PitPolicy]
    evaluation: Optional[PolicyEvaluation]
    field_evaluation: Optional[PolicyEvaluation]
    screen_gain_pts: float
    screen_fire_rate: float
    n_screen: int
    significant: bool
    rule: str
    notes: List[str]


def search_policy(
    ctx: PolicyContext,
    n_screen: int = 800,
    n_eval: int = 3000,
    seed: int = 0,
    field_sensitivity: bool = True,
    n_field: int = 1500,
) -> PolicySearchResult:
    """
    Search (see module docstring) on seed `seed`, evaluate the winner on a fresh seed `seed + 1`. `significant`
    means the 95% interval of the points gain on the fresh seed excludes 0 on the positive side.
    """
    tables = build_event_tables(ctx, n_screen, seed)
    pol, screen_gain, screen_rate = best_policy_from_tables(tables, ctx.total_laps)
    notes: List[str] = []
    plan = ctx.hero_plan
    if pol is None or screen_rate == 0.0:
        return PolicySearchResult(None, None, None, 0.0, 0.0, n_screen, False,
                                  "No safety-car reaction was possible for this plan (no event could trigger an allowed stop).",
                                  ["no eligible SC/VSC events in the screening sample"])
    ev = evaluate_policy(ctx, pol, n_eval, seed + 1)
    fev = evaluate_policy(ctx, pol, n_field, seed + 1, field_reacts=True) if field_sensitivity else None
    signif = ev.gain_pts - ev.gain_pts_ci > 0
    if screen_gain > ev.gain_pts + 2 * ev.gain_pts_ci:
        notes.append("the screening gain was much higher than the fresh-seed gain: selection effect, trust the fresh-seed number")
    if signif:
        rule = pol.describe(plan)
    else:
        rule = ("No safety-car rule beats the fixed plan by more than the simulation noise. Best candidate: "
                + pol.describe(plan))
    return PolicySearchResult(pol, ev, fev, screen_gain, screen_rate, n_screen, signif, rule, notes)


# --------------------------------------------------------------------------- high level


def build_context(
    track_id: str,
    tire_model: dict,
    driver: Optional[str] = None,
    base_plan: Optional[Sequence[Stint]] = None,
    rnd: Optional[int] = None,
    grid: Optional[Dict[str, int]] = None,
    safety_cars: bool = True,
    seed: int = 0,
    total_laps: Optional[int] = None,
) -> PolicyContext:
    """
    Field and hero for a track. Default hero: the driver on pole (grid slot 1) of the round at the track;
    default base plan: the optimizer's #1 plan (no per-car jitter). Every other car keeps the grid sim's
    default plan mix. Raises ValueError for an unknown driver or a plan that does not cover the race.
    """
    from app.core import grid_inputs as G
    from app.core.race_grid_sim import load_interaction_params
    from app.core.safety_cars import load_params

    gi = G.build_inputs(track_id, tire_model, rnd=rnd, grid=grid, total_laps=total_laps, seed=seed)
    codes = [c.code for c in gi.cars]
    if driver:
        driver = driver.upper()
        if driver not in codes:
            raise ValueError(f"driver {driver!r} is not on the grid: {codes}")
        hero = codes.index(driver)
    else:
        hero = int(np.argmin([c.grid for c in gi.cars]))
    plan = (
        [Stint(s.compound, s.laps, s.start_age) for s in base_plan]
        if base_plan
        else G.top_plans(gi.track, tire_model, gi.total_laps)[0]
    )
    if sum(s.laps for s in plan) != gi.total_laps:
        raise ValueError(f"base plan covers {sum(s.laps for s in plan)} laps, the race is {gi.total_laps}")
    cars = list(gi.cars)
    cars[hero] = replace(cars[hero], strategy=plan)
    return PolicyContext(
        cars=cars, hero=hero, base_laptime_s=gi.track["base_laptime_s"], tire_model=tire_model,
        total_laps=gi.total_laps, pit_loss_s=gi.track["pit_loss_s"], caps=stint_caps(gi.track, gi.total_laps),
        safety_cars=load_params() if safety_cars else None, interaction=load_interaction_params(), track_id=track_id,
    )
