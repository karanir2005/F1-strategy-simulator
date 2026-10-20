"""
Stint-plan optimizer (early Phase 4): deterministic screen + CRN Monte Carlo.

Pipeline
  1. Enumerate plans: 1..max_stops stops, compound sequences with >= 2 distinct
     dry compounds (the FIA two-compound rule), every stint >= MIN_STINT_LAPS
     and <= that compound's observed cap (max_stint_laps -- never extrapolate
     past observed data), stint lengths on a coarse grid.
  2. Screen every plan with a closed-form deterministic total. This is the same
     arithmetic as race_simulator.simulate_race (tyre delta + compound offset +
     fuel + pit loss on each out-lap), but with per-compound cumulative sums so a
     stint costs O(1) and tens of thousands of plans screen in well under a
     second. test_optimizer.py asserts the two agree.
  3. Refine: hill-climb the best plan of each compound sequence by moving laps
     between stints (+-1, +-2) to undo the grid coarseness.
  4. Run the best MAX_FINALISTS plans through monte_carlo.simulate_totals (common
     random numbers, so ranks and paired deltas are consistent) and rank by mean.

Everything is deterministic given the seed. Not DP/GA yet (Roadmap Phase 4
proper); this is the screen-then-simulate version.
"""

import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.core.monte_carlo import (
    DEFAULT_LAP_TIME_NOISE_STD,
    DEFAULT_PIT_LOSS_NOISE_STD,
    simulate_totals,
)
from app.core.race_simulator import Stint
from app.core.tire_model import compound_pace_offset, fuel_effect, predict_delta

# Minimum stint length applies to EVERY stint (first and last included), so a plan cannot
# dodge wear with a token 5-lap stint. Floor of 8 laps, or 10% of race distance if larger:
# real stops almost never come before ~lap 8-10 (the shortest stint among 2025 winners'
# plans was ~7-10 laps, e.g. China M10, Qatar M7 under the tyre-life rule), and a lap-5 stop
# is a model artefact of the cap floor + cheap global degradation (T-019 finding 2).
# Clamped to total_laps // 2 so very short test distances stay feasible.
MIN_STINT_LAPS = 8
MIN_STINT_FRACTION = 0.10


def min_stint_laps(total_laps: int) -> int:
    return max(1, min(max(MIN_STINT_LAPS, round(MIN_STINT_FRACTION * total_laps)), total_laps // 2))


MAX_FINALISTS = 20
# how many compound sequences (best grid plan each) get hill-climbed
MAX_REFINED_SEQUENCES = 30
# floor on observed caps: some tracks only saw short green stints for a compound (e.g. 8-12 laps)
MIN_STINT_CAP = 15

ASSUMPTIONS = [
    "Single car in clean air: no traffic, no opponents, no undercut/overcut interaction.",
    "No safety cars, virtual safety cars, red flags or weather; dry compounds only.",
    "Global tyre-wear curves fitted across all pulled races, not per-track degradation.",
    "Compound pace offsets are fitted (with a forced ordering); cliff penalties are literature-ballpark placeholders "
    "(tire_model.COMPOUND_PACE_OFFSET_PLACEHOLDER / COMPOUND_CLIFF_PENALTY_PLACEHOLDER), "
    "so SOFT-vs-HARD trade-offs are indicative, not fitted.",
    "Stints are capped at the p90 of observed stint lengths for that compound at this track "
    "(typical_stint_laps, never above the observed maximum), or the observed maximum when fewer than 5 "
    "stints were seen, with a floor of 15 laps because some compounds only had short observed "
    "stints; plans needing longer stints are not considered.",
    "Every stint (first and last included) is at least max(8 laps, 10% of race distance); real teams do not stop on lap 5.",
    "Plans that are the same stints in a different order are shown once (the best-ordered one).",
    "All stints start on new tyres; at least 2 distinct dry compounds are used (two-compound rule).",
    "Gaps between plans are model gaps (paired, common random numbers); replay error vs real "
    "races is tens of seconds, so differences of a few seconds are not meaningful.",
]


@dataclass
class Plan:
    compounds: Tuple[str, ...]
    laps: Tuple[int, ...]
    det_total: float


@dataclass
class OptimizeResult:
    plans: List[dict]
    n_candidates_screened: int
    n_candidates_simulated: int
    effective_caps: Dict[str, int]  # per-compound stint cap actually used (after floor + clip)
    min_stint_laps: int


class NoFeasiblePlan(ValueError):
    pass


class _Cost:
    """O(1) deterministic stint costs, consistent with simulate_race."""

    def __init__(self, tire_model, compounds, caps, total_laps, base_laptime, pit_loss):
        self.total_laps = total_laps
        self.base = base_laptime
        self.pit_loss = pit_loss
        # tire_cum[c][n] = seconds of tyre delta + pace offset over ages 1..n
        self.tire_cum = {}
        for c in compounds:
            off = compound_pace_offset(tire_model, c)
            deltas = [predict_delta(tire_model, c, age) + off for age in range(1, caps[c] + 1)]
            self.tire_cum[c] = np.concatenate([[0.0], np.cumsum(deltas)])
        # fuel_cum[l] = sum of fuel_effect over laps 1..l
        fuel = [fuel_effect(tire_model, lap, total_laps) for lap in range(1, total_laps + 1)]
        self.fuel_cum = np.concatenate([[0.0], np.cumsum(fuel)])

    def total(self, compounds: Sequence[str], laps: Sequence[int]) -> float:
        t = self.base * self.total_laps + self.pit_loss * (len(laps) - 1)
        start = 0
        for c, n in zip(compounds, laps):
            t += self.tire_cum[c][n] + self.fuel_cum[start + n] - self.fuel_cum[start]
            start += n
        return float(t)


def _length_grid(n_stints: int, total: int, caps_max: int, step: int, min_lap: int) -> np.ndarray:
    """All (n_stints,) length tuples on the grid; the last stint takes the remainder."""
    values = list(range(min_lap, min(caps_max, total) + 1, step))
    rows = []
    for head in itertools.product(values, repeat=n_stints - 1):
        last = total - sum(head)
        if min_lap <= last <= caps_max:
            rows.append(head + (last,))
    return np.array(rows, dtype=np.int64).reshape(-1, n_stints)


def _grid_step(n_stints: int) -> int:
    return {2: 1, 3: 2}.get(n_stints, 3)


def _refine(cost: _Cost, compounds, laps, caps, min_lap) -> Plan:
    laps = list(laps)
    best = cost.total(compounds, laps)
    improved = True
    while improved:
        improved = False
        for i, j in itertools.permutations(range(len(laps)), 2):
            for d in (1, 2):
                cand = list(laps)
                cand[i] -= d
                cand[j] += d
                if cand[i] < min_lap or cand[j] > caps[compounds[j]]:
                    continue
                t = cost.total(compounds, cand)
                if t < best - 1e-9:
                    best, laps, improved = t, cand, True
    return Plan(tuple(compounds), tuple(laps), best)


def optimize(
    tire_model: dict,
    total_laps: int,
    pit_loss_s: float,
    base_laptime_s: float,
    max_stint_laps: Dict[str, int],
    max_stops: int = 3,
    n_sims: int = 2000,
    top_n: int = 5,
    seed: Optional[int] = 0,
    pit_loss_noise_std: float = DEFAULT_PIT_LOSS_NOISE_STD,
    lap_time_noise_std: float = DEFAULT_LAP_TIME_NOISE_STD,
    typical_stint_laps: Optional[Dict[str, Optional[int]]] = None,
) -> OptimizeResult:
    compounds = [c for c in ("SOFT", "MEDIUM", "HARD") if c in max_stint_laps and c in tire_model]
    if len(compounds) < 2:
        raise NoFeasiblePlan("need caps for at least 2 dry compounds to satisfy the two-compound rule")
    # robust cap: the observed max is one outlier stint away from unrealistic, so use the
    # p90 ("typical") when the track has one; then floor and clip as before.
    raw = dict(max_stint_laps)
    for c, v in (typical_stint_laps or {}).items():
        if v is not None and c in raw:
            raw[c] = min(int(raw[c]), int(v))
    caps = {c: min(max(int(raw[c]), MIN_STINT_CAP), total_laps) for c in compounds}
    min_lap = min_stint_laps(total_laps)
    cost = _Cost(tire_model, compounds, caps, total_laps, base_laptime_s, pit_loss_s)

    best_per_seq: Dict[Tuple[str, ...], Plan] = {}
    n_screened = 0
    for n_stints in range(2, max_stops + 2):
        if n_stints * min_lap > total_laps:
            break
        grid = _length_grid(n_stints, total_laps, max(caps.values()), _grid_step(n_stints), min_lap)
        if not len(grid):
            continue
        starts = np.cumsum(grid, axis=1) - grid  # laps already run before each stint
        for seq in itertools.product(compounds, repeat=n_stints):
            if len(set(seq)) < 2:
                continue
            mask = np.ones(len(grid), dtype=bool)
            for k, c in enumerate(seq):
                mask &= grid[:, k] <= caps[c]
            if not mask.any():
                continue
            g, st = grid[mask], starts[mask]
            tot = np.full(len(g), base_laptime_s * total_laps + pit_loss_s * (n_stints - 1))
            for k, c in enumerate(seq):
                tot += cost.tire_cum[c][g[:, k]] + cost.fuel_cum[st[:, k] + g[:, k]] - cost.fuel_cum[st[:, k]]
            n_screened += len(g)
            i = int(np.argmin(tot))
            best_per_seq[seq] = Plan(seq, tuple(int(x) for x in g[i]), float(tot[i]))
    if not best_per_seq:
        raise NoFeasiblePlan(
            "no plan satisfies the two-compound rule, minimum stint length and max_stint_laps "
            f"caps {caps} for {total_laps} laps with up to {max_stops} stops"
        )

    seeds = sorted(best_per_seq.values(), key=lambda p: (p.det_total, p.compounds))[:MAX_REFINED_SEQUENCES]
    refined = [_refine(cost, p.compounds, p.laps, caps, min_lap) for p in seeds]
    # same stints in a different order differ only by a fuel-timing sliver: keep the best order
    finalists, seen = [], set()
    for p in sorted(refined, key=lambda p: (p.det_total, p.compounds, p.laps)):
        key = tuple(sorted(zip(p.compounds, p.laps)))
        if key not in seen:
            seen.add(key)
            finalists.append(p)
    finalists = finalists[:MAX_FINALISTS]

    named = {str(i): [Stint(c, n) for c, n in zip(p.compounds, p.laps)] for i, p in enumerate(finalists)}
    totals = simulate_totals(
        named,
        base_laptime_s,
        tire_model,
        n_simulations=n_sims,
        lap_time_noise_std=lap_time_noise_std,
        pit_loss_noise_std=pit_loss_noise_std,
        pit_loss_seconds=pit_loss_s,
        seed=seed,
    )
    stacked = np.vstack([totals[str(i)] for i in range(len(finalists))])
    means = stacked.mean(axis=1)
    order = np.argsort(means, kind="stable")
    best = stacked[order[0]]

    plans = []
    for rank, idx in enumerate(order[:top_n], start=1):
        p = finalists[int(idx)]
        arr = stacked[idx]
        delta = arr - best
        d10, d90 = np.percentile(delta, [10, 90])
        p10, p90 = np.percentile(arr, [10, 90])
        plans.append(
            {
                "rank": rank,
                "n_stops": len(p.laps) - 1,
                "stints": [{"compound": c, "laps": int(n), "start_age": 1} for c, n in zip(p.compounds, p.laps)],
                # last lap of each stint but the final (pit at the end of that lap)
                "pit_laps": [int(x) for x in np.cumsum(p.laps)[:-1]],
                "mean_s": float(means[idx]),
                "p10_s": float(p10),
                "p90_s": float(p90),
                "delta_to_best_mean_s": float(delta.mean()),
                "delta_to_best_p10_s": float(d10),
                "delta_to_best_p90_s": float(d90),
            }
        )
    return OptimizeResult(plans, n_screened, len(finalists), dict(caps), min_lap)
