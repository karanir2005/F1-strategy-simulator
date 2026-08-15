import numpy as np
import pytest

from app.core.monte_carlo import compare_strategies, run_monte_carlo
from app.core.race_simulator import Stint


def test_run_monte_carlo_returns_n_samples(toy_tire_model):
    strategy = [Stint("MEDIUM", 20)]
    result = run_monte_carlo(
        strategy, base_laptime=90.0, tire_model=toy_tire_model, n_simulations=50, seed=1
    )
    assert len(result.total_times) == 50


def test_run_monte_carlo_is_reproducible_given_a_seed(toy_tire_model):
    strategy = [Stint("MEDIUM", 20)]
    r1 = run_monte_carlo(
        strategy, base_laptime=90.0, tire_model=toy_tire_model, n_simulations=50, seed=7
    )
    r2 = run_monte_carlo(
        strategy, base_laptime=90.0, tire_model=toy_tire_model, n_simulations=50, seed=7
    )
    assert np.array_equal(r1.total_times, r2.total_times)


def test_percentiles_are_ordered(toy_tire_model):
    strategy = [Stint("MEDIUM", 20)]
    result = run_monte_carlo(
        strategy, base_laptime=90.0, tire_model=toy_tire_model, n_simulations=200, seed=2
    )
    assert result.percentile(10) <= result.percentile(50) <= result.percentile(90)


def test_compare_strategies_win_probabilities_sum_to_one(toy_tire_model):
    strategies = {
        "A": [Stint("MEDIUM", 20)],
        "B": [Stint("SOFT", 10), Stint("HARD", 10)],
    }
    summary = compare_strategies(
        strategies, base_laptime=90.0, tire_model=toy_tire_model, n_simulations=200, seed=3
    )
    total = sum(s["win_probability"] for s in summary.values())
    assert total == pytest.approx(1.0)


def test_strategy_with_unnecessary_extra_pit_stop_loses_more_often(toy_tire_model):
    """
    With noise small relative to the ~22s pit stop cost, a strategy that
    pits without any real strategic benefit (same compound, split into
    two stints) should lose to the equivalent one-stint strategy almost
    every time -- a sanity check that the pit-loss cost is actually
    driving comparisons, not swamped by noise or a modeling bug.
    """
    strategies = {
        "no_pit": [Stint("MEDIUM", 20)],
        "unnecessary_pit": [Stint("MEDIUM", 10), Stint("MEDIUM", 10)],
    }
    summary = compare_strategies(
        strategies,
        base_laptime=90.0,
        tire_model=toy_tire_model,
        n_simulations=200,
        lap_time_noise_std=0.01,
        pit_loss_noise_std=0.01,
        pit_loss_seconds=22.0,
        seed=4,
    )
    assert summary["no_pit"]["win_probability"] > summary["unnecessary_pit"]["win_probability"]
