"""
Monte Carlo layer: runs a strategy (or several, for comparison) through
race_simulator.py thousands of times with randomized lap-time and pit-stop
noise, then summarizes the resulting distribution of race times.

SCOPE NOTE: this simulates one car's strategy in isolation, not a full
20-car grid -- there's no opponent modeling yet. "win_probability" here
means "probability this strategy beats the OTHER STRATEGIES PASSED IN TO
compare_strategies", not "probability of winning the actual race" (that
would need per-driver pace and a modeled field of opponents, which is a
bigger future step, not this module).
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
    rng = np.random.default_rng(seed)
    total_times = np.empty(n_simulations)
    for i in range(n_simulations):
        result = simulate_race(
            strategy,
            base_laptime,
            tire_model,
            pit_loss_seconds=pit_loss_seconds,
            lap_time_noise_std=lap_time_noise_std,
            pit_loss_noise_std=pit_loss_noise_std,
            rng=rng,
        )
        total_times[i] = result.total_time
    return MonteCarloResult(strategy_name=strategy_name, total_times=total_times)


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
    per-strategy stats (mean/std/p10/p50/p90) plus win_probability -- see
    module docstring for what that probability is (and isn't) relative to.
    """
    rng = np.random.default_rng(seed)
    names = list(strategies.keys())
    times = {name: np.empty(n_simulations) for name in names}

    for i in range(n_simulations):
        for name in names:
            result = simulate_race(
                strategies[name],
                base_laptime,
                tire_model,
                pit_loss_seconds=pit_loss_seconds,
                lap_time_noise_std=lap_time_noise_std,
                pit_loss_noise_std=pit_loss_noise_std,
                rng=rng,
            )
            times[name][i] = result.total_time

    wins = {name: 0 for name in names}
    for i in range(n_simulations):
        winner = min(names, key=lambda n: times[n][i])
        wins[winner] += 1

    summary = {}
    for name in names:
        arr = times[name]
        summary[name] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "win_probability": wins[name] / n_simulations,
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
    print(f"\n({DEFAULT_N_SIMULATIONS} sims/strategy in {elapsed:.1f}s)")
