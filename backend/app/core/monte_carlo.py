"""
Monte Carlo layer: runs a strategy (or several, for comparison) through
race_simulator.py thousands of times with randomized lap-time and pit-stop
noise, then summarizes the resulting distribution of race times.

All simulations are evaluated at once with numpy (see simulate_totals),
and strategies being compared share common random numbers (same draws
for every strategy within an iteration).

SCOPE NOTE: this simulates one car's strategy in isolation, not a full
20-car grid -- there's no opponent modeling yet. "win_probability" here
means "probability this strategy beats the OTHER STRATEGIES PASSED IN TO
compare_strategies", not "probability of winning the actual race" (that
would need per-driver pace and a modeled field of opponents, which is a
bigger future step, not this module).

CAVEATS that apply to every number compare_strategies returns (read
these before putting a percentage in front of a user):
  - ONE CAR IN ISOLATION: no traffic, no safety cars, no opponents, no
    reactive pit calls. Strategies only interact with the random draws,
    never with each other.
  - ADDITIVE NOISE ONLY: lap-time and pit-loss noise are iid Gaussian
    terms added on top of a deterministic race; they never feed back
    into degradation, fuel or pit timing (see simulate_totals).
  - COMMON RANDOM NUMBERS: every strategy in a comparison sees the same
    draws (lap j shares lap j's draw, the k-th stop shares the k-th pit
    draw). Combined with additive noise, two strategies of the same race
    length and stop count differ by the SAME constant in every
    iteration, so their win_probability is 0 or 1 regardless of how
    small that gap is (a 0.3s edge reads as ~100%). Only pit draws for
    stops one strategy makes and the other doesn't add any spread.
    win_probability is therefore close to "which mean is lowest" and
    carries little uncertainty information under the current model.
  - The paired delta (delta_mean / delta_p10 / delta_p90 from
    compare_strategies: this strategy's total minus the best-by-mean
    strategy's total, iteration by iteration) is the honest headline:
    it shows how big the gap is, not just who is ahead. Even so, it is
    the gap *under this model*; validate_replay.py errors against real
    races are tens of seconds, so sub-second deltas are not meaningful
    differences.
  - Comparisons are only meaningful between strategies of the same
    total race laps (a shorter race always "wins" under CRN), so
    compare_strategies rejects unequal totals. simulate_totals, the
    low-level primitive, does not check.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.core.race_simulator import DEFAULT_PIT_LOSS_SECONDS, Stint, simulate_race

DEFAULT_N_SIMULATIONS = 10_000

# Calibrated from real data via calibrate_lap_noise.py / calibrate_pit_loss.py
# (see those scripts' docstrings for method). Both use a robust estimate
# rather than the raw pooled std, because the raw residuals are
# heavy-tailed (lapped traffic, minor yellow-flag contamination
# is_green_flag doesn't fully filter, etc.) and a single Gaussian noise
# term shouldn't be stretched to cover those rare large deviations:
#   - lap time: MAD-based robust std of per-lap residuals (actual minus
#     race_simulator.py's predicted lap time) after removing each
#     driver's average bias, pooled across all 10 races (raw std 2.76s,
#     robust std 0.77s -- used the latter)
#   - pit loss: pooled std of real pit-stop durations, excluding Monaco
#     (whose narrow pit lane makes it a genuine outlier, not noise --
#     see race_simulator.py's DEFAULT_PIT_LOSS_SECONDS comment)
DEFAULT_LAP_TIME_NOISE_STD = 0.8  # seconds, per lap
DEFAULT_PIT_LOSS_NOISE_STD = 5.0  # seconds, per pit stop


@dataclass
class MonteCarloResult:
    strategy_name: str
    total_times: np.ndarray

    @property
    def mean(self) -> float:
        return float(np.mean(self.total_times))

    @property
    def std(self) -> float:
        return float(np.std(self.total_times))

    def percentile(self, p: float) -> float:
        return float(np.percentile(self.total_times, p))


def _draw_noise(
    rng: np.random.Generator,
    n_simulations: int,
    n_columns: int,
    std: float,
) -> np.ndarray:
    """
    (n_simulations, n_columns) Gaussian noise matrix, or exact zeros when
    std == 0 (so the noise-off path consumes no random draws and adds
    exactly nothing to the deterministic total).
    """
    if std and n_columns:
        return rng.normal(0.0, std, size=(n_simulations, n_columns))
    return np.zeros((n_simulations, n_columns))


def simulate_totals(
    strategies: Dict[str, List[Stint]],
    base_laptime: float,
    tire_model: dict,
    n_simulations: int = DEFAULT_N_SIMULATIONS,
    lap_time_noise_std: float = DEFAULT_LAP_TIME_NOISE_STD,
    pit_loss_noise_std: float = DEFAULT_PIT_LOSS_NOISE_STD,
    pit_loss_seconds: float = DEFAULT_PIT_LOSS_SECONDS,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Vectorized Monte Carlo core: returns {name: array of n_simulations
    total race times} for every strategy, using common random numbers.

    VECTORIZATION: in race_simulator.py's model both noise terms are
    purely additive -- a lap's noise doesn't feed back into degradation,
    fuel or pit timing. So a noisy total is exactly
        deterministic total + sum(lap noise) + sum(pit noise)
    and the deterministic part only needs computing ONCE per strategy
    (via simulate_race, which stays the single-run reference), with the
    noise summed across all simulations at once in numpy. If the engine
    ever gets noise that interacts with state (e.g. random safety cars
    that change pit timing), this decomposition stops being valid and
    must be revisited rather than silently reused.

    COMMON RANDOM NUMBERS: one (n_simulations, max_laps) lap-noise matrix
    and one (n_simulations, max_stops) pit-noise matrix are drawn per
    call and SHARED by every strategy. Within an iteration, lap index j
    gets the same noise draw in every strategy and the k-th pit stop gets
    the same pit-noise draw in every strategy. Differences between
    strategies then reflect the strategies, not which one happened to
    get luckier draws -- the standard variance-reduction choice for
    comparing alternatives under simulation. Consequences worth knowing:
      - for strategies of the same race length, lap noise cancels out of
        every pairwise comparison entirely; only pit-noise draws for
        stops the other strategy doesn't make can flip a ranking. This is
        the honest result of the current model (iid symmetric lap noise
        carries no strategy information -- team/ROADMAP.md point 3), not
        a bug; it changes once noise interacts with race state.
      - two identical strategies produce identical totals, so a tie is
        broken toward the strategy listed first (see compare_strategies).
    Draw order is lap noise first, then pit noise, from one Generator
    seeded by `seed` -- the same seed reproduces the same results.
    """
    rng = np.random.default_rng(seed)

    deterministic = {
        name: simulate_race(
            stints, base_laptime, tire_model, pit_loss_seconds=pit_loss_seconds
        )
        for name, stints in strategies.items()
    }
    max_laps = max((len(r.lap_times) for r in deterministic.values()), default=0)
    max_stops = max((len(r.pit_laps) for r in deterministic.values()), default=0)

    # cumulative sums so "noise summed over the first n laps / stops" is a
    # single column lookup per strategy; a leading zero column handles
    # n == 0 (e.g. a no-stop strategy) without special-casing
    lap_noise = _draw_noise(rng, n_simulations, max_laps, lap_time_noise_std)
    pit_noise = _draw_noise(rng, n_simulations, max_stops, pit_loss_noise_std)
    zeros = np.zeros((n_simulations, 1))
    lap_cum = np.hstack([zeros, np.cumsum(lap_noise, axis=1)])
    pit_cum = np.hstack([zeros, np.cumsum(pit_noise, axis=1)])

    return {
        name: result.total_time
        + lap_cum[:, len(result.lap_times)]
        + pit_cum[:, len(result.pit_laps)]
        for name, result in deterministic.items()
    }


def run_monte_carlo(
    strategy: List[Stint],
    base_laptime: float,
    tire_model: dict,
    n_simulations: int = DEFAULT_N_SIMULATIONS,
    lap_time_noise_std: float = DEFAULT_LAP_TIME_NOISE_STD,
    pit_loss_noise_std: float = DEFAULT_PIT_LOSS_NOISE_STD,
    pit_loss_seconds: float = DEFAULT_PIT_LOSS_SECONDS,
    seed: Optional[int] = None,
    strategy_name: str = "strategy",
) -> MonteCarloResult:
    """Run one strategy n_simulations times with randomized noise; return the resulting time distribution."""
    totals = simulate_totals(
        {strategy_name: strategy},
        base_laptime,
        tire_model,
        n_simulations=n_simulations,
        lap_time_noise_std=lap_time_noise_std,
        pit_loss_noise_std=pit_loss_noise_std,
        pit_loss_seconds=pit_loss_seconds,
        seed=seed,
    )
    return MonteCarloResult(strategy_name=strategy_name, total_times=totals[strategy_name])


def compare_strategies(
    strategies: Dict[str, List[Stint]],
    base_laptime: float,
    tire_model: dict,
    n_simulations: int = DEFAULT_N_SIMULATIONS,
    lap_time_noise_std: float = DEFAULT_LAP_TIME_NOISE_STD,
    pit_loss_noise_std: float = DEFAULT_PIT_LOSS_NOISE_STD,
    pit_loss_seconds: float = DEFAULT_PIT_LOSS_SECONDS,
    seed: Optional[int] = None,
) -> Dict[str, dict]:
    """
    Run several candidate strategies through Monte Carlo, and on each
    iteration record which one had the lowest total time. Returns
    {name: stats} where stats has:
      - mean / std / p10 / p50 / p90: the strategy's own total race time
        distribution, seconds
      - win_probability: share of iterations in which this strategy had
        the lowest total (exact ties go to the strategy listed first)
      - delta_mean / delta_p10 / delta_p90: PAIRED comparison against the
        reference strategy, seconds. For each iteration i,
        delta[i] = total[i] - reference_total[i] (same draws, thanks to
        CRN); these are the mean and 10th/90th percentiles of that
        array. Positive = slower than the reference. The reference's own
        deltas are exactly 0.
      - delta_reference: name of the reference strategy -- the one with
        the lowest MEAN total (ties -> first listed). Same for every
        entry.

    Caveats (also in the module docstring, read it): one car in
    isolation, additive noise only, and common random numbers -- all
    strategies share the same draws within an iteration (see
    simulate_totals for exactly what is shared). Under that model,
    same-length same-stop-count strategies differ by a constant every
    iteration, so win_probability collapses to 0/1 even for a 0.3s gap.
    Prefer the paired delta as the headline number: it says how large
    the gap is, and its p10-p90 spread shows how much of it pit-stop
    noise can swing.

    Raises ValueError if the strategies don't all cover the same total
    number of laps: under CRN the shorter race shares the longer one's
    first-N lap draws and simply skips the rest, so it "wins" every
    iteration -- a meaningless comparison. (run_monte_carlo, for a single
    strategy, has no such constraint.)
    """
    total_laps = {name: sum(s.laps for s in stints) for name, stints in strategies.items()}
    if len(set(total_laps.values())) > 1:
        raise ValueError(
            "compare_strategies needs every strategy to cover the same total "
            f"number of laps, got {total_laps}"
        )

    times = simulate_totals(
        strategies,
        base_laptime,
        tire_model,
        n_simulations=n_simulations,
        lap_time_noise_std=lap_time_noise_std,
        pit_loss_noise_std=pit_loss_noise_std,
        pit_loss_seconds=pit_loss_seconds,
        seed=seed,
    )
    names = list(strategies.keys())
    stacked = np.vstack([times[name] for name in names])  # (n_strategies, n_sims)

    # np.argmin returns the first minimum, matching the old
    # min(names, key=...) tie-break toward the earlier-listed strategy
    winners = np.argmin(stacked, axis=0)
    wins = np.bincount(winners, minlength=len(names))

    # reference for the paired delta = best strategy by mean (same
    # first-listed tie-break). Paired, not a difference of the two
    # marginal distributions: subtracting row-by-row keeps the CRN
    # pairing, so lap noise both strategies share cancels exactly and
    # what's left is the strategy effect plus only the unshared draws.
    means = stacked.mean(axis=1)
    ref_idx = int(np.argmin(means))
    deltas = stacked - stacked[ref_idx]

    summary = {}
    for idx, name in enumerate(names):
        arr = stacked[idx]
        p10, p50, p90 = np.percentile(arr, [10, 50, 90])
        d10, d90 = np.percentile(deltas[idx], [10, 90])
        summary[name] = {
            "mean": float(means[idx]),
            "std": float(np.std(arr)),
            "p10": float(p10),
            "p50": float(p50),
            "p90": float(p90),
            "win_probability": int(wins[idx]) / n_simulations,
            "delta_mean": float(np.mean(deltas[idx])),
            "delta_p10": float(d10),
            "delta_p90": float(d90),
            "delta_reference": names[ref_idx],
        }
    return summary


if __name__ == "__main__":
    import time

    from app.core.tire_model import build_tire_model

    tire_model = build_tire_model()

    strategies = {
        "1-stop (M25/H32)": [Stint("MEDIUM", 25), Stint("HARD", 32)],
        "2-stop (S15/M20/H22)": [
            Stint("SOFT", 15),
            Stint("MEDIUM", 20),
            Stint("HARD", 22),
        ],
    }

    start = time.time()
    summary = compare_strategies(
        strategies, base_laptime=90.0, tire_model=tire_model, seed=42
    )
    elapsed = time.time() - start

    for name, stats in summary.items():
        print(f"{name}:")
        print(f"  mean={stats['mean']:.1f}s  std={stats['std']:.2f}s")
        print(
            f"  p10={stats['p10']:.1f}s  p50={stats['p50']:.1f}s  p90={stats['p90']:.1f}s"
        )
        print(f"  win_probability={stats['win_probability']:.1%}")
    print(f"\n({DEFAULT_N_SIMULATIONS} sims/strategy in {elapsed:.3f}s)")
