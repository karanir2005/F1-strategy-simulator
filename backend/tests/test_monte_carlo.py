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


# --- T-004: vectorized Monte Carlo + common random numbers ---

from app.core.monte_carlo import simulate_totals  # noqa: E402
from app.core.race_simulator import simulate_race  # noqa: E402

STRATEGIES = {
    "1-stop": [Stint("MEDIUM", 12), Stint("HARD", 13)],
    "2-stop": [Stint("SOFT", 8), Stint("MEDIUM", 8, start_age=3), Stint("HARD", 9)],
    "no-stop": [Stint("HARD", 25)],
}


def test_noise_off_vectorized_path_reproduces_simulate_race(toy_tire_model):
    totals = simulate_totals(
        STRATEGIES,
        base_laptime=90.0,
        tire_model=toy_tire_model,
        n_simulations=20,
        lap_time_noise_std=0.0,
        pit_loss_noise_std=0.0,
        pit_loss_seconds=22.0,
        seed=1,
    )
    for name, stints in STRATEGIES.items():
        reference = simulate_race(
            stints, base_laptime=90.0, tire_model=toy_tire_model, pit_loss_seconds=22.0
        ).total_time
        assert np.allclose(totals[name], reference, rtol=0, atol=1e-9)


def test_noise_off_compare_strategies_matches_simulate_race(toy_tire_model):
    summary = compare_strategies(
        STRATEGIES,
        base_laptime=90.0,
        tire_model=toy_tire_model,
        n_simulations=10,
        lap_time_noise_std=0.0,
        pit_loss_noise_std=0.0,
        seed=2,
    )
    for name, stints in STRATEGIES.items():
        reference = simulate_race(stints, base_laptime=90.0, tire_model=toy_tire_model).total_time
        assert summary[name]["mean"] == pytest.approx(reference, abs=1e-9)
        assert summary[name]["std"] == pytest.approx(0.0, abs=1e-9)


def test_noisy_vectorized_path_matches_a_per_lap_reference_loop(toy_tire_model):
    """
    Independent check of the noise decomposition: rebuild each total by
    hand from simulate_race's noise-free lap trace plus the SAME noise
    draws (lap j -> column j, k-th stop -> pit column k) and compare.
    """
    n, lap_std, pit_std, seed = 30, 0.8, 5.0, 11
    totals = simulate_totals(
        STRATEGIES,
        base_laptime=90.0,
        tire_model=toy_tire_model,
        n_simulations=n,
        lap_time_noise_std=lap_std,
        pit_loss_noise_std=pit_std,
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    lap_noise = rng.normal(0.0, lap_std, size=(n, 25))
    pit_noise = rng.normal(0.0, pit_std, size=(n, 2))
    for name, stints in STRATEGIES.items():
        trace = simulate_race(stints, base_laptime=90.0, tire_model=toy_tire_model)
        for i in range(n):
            expected = sum(
                lap + lap_noise[i, j] for j, lap in enumerate(trace.lap_times)
            ) + sum(pit_noise[i, k] for k in range(len(trace.pit_laps)))
            assert totals[name][i] == pytest.approx(expected)


def test_common_random_numbers_identical_strategies_get_identical_draws(toy_tire_model):
    strategies = {
        "A": [Stint("MEDIUM", 10), Stint("HARD", 10)],
        "B": [Stint("MEDIUM", 10), Stint("HARD", 10)],
    }
    totals = simulate_totals(
        strategies, base_laptime=90.0, tire_model=toy_tire_model, n_simulations=500, seed=5
    )
    assert np.array_equal(totals["A"], totals["B"])
    # exact ties go to the strategy listed first
    summary = compare_strategies(
        strategies, base_laptime=90.0, tire_model=toy_tire_model, n_simulations=500, seed=5
    )
    assert summary["A"]["win_probability"] == 1.0


def test_common_random_numbers_noise_cancels_for_same_length_same_stops(toy_tire_model):
    """
    Same race length and same number of stops -> every lap draw and every
    pit draw is shared, so the per-iteration difference is exactly the
    deterministic difference, regardless of noise size.
    """
    a = [Stint("MEDIUM", 10), Stint("HARD", 10)]
    b = [Stint("SOFT", 8), Stint("HARD", 12)]
    totals = simulate_totals(
        {"a": a, "b": b},
        base_laptime=90.0,
        tire_model=toy_tire_model,
        n_simulations=500,
        lap_time_noise_std=3.0,
        pit_loss_noise_std=5.0,
        seed=9,
    )
    det_diff = (
        simulate_race(a, base_laptime=90.0, tire_model=toy_tire_model).total_time
        - simulate_race(b, base_laptime=90.0, tire_model=toy_tire_model).total_time
    )
    assert np.allclose(totals["a"] - totals["b"], det_diff, atol=1e-6)


def test_common_random_numbers_do_not_depend_on_which_strategies_are_compared(toy_tire_model):
    """
    A strategy's draws come from fixed lap/stop columns, so adding a
    competitor that doesn't widen the noise matrices (no more laps or
    stops than the existing ones) must not change its sampled totals.
    """
    base = {"1-stop": STRATEGIES["1-stop"], "2-stop": STRATEGIES["2-stop"]}
    with_extra = dict(base, **{"no-stop": STRATEGIES["no-stop"]})
    kwargs = dict(base_laptime=90.0, tire_model=toy_tire_model, n_simulations=200, seed=3)
    t1 = simulate_totals(base, **kwargs)
    t2 = simulate_totals(with_extra, **kwargs)
    assert np.array_equal(t1["1-stop"], t2["1-stop"])
    assert np.array_equal(t1["2-stop"], t2["2-stop"])


def test_compare_strategies_is_reproducible_given_a_seed(toy_tire_model):
    kwargs = dict(base_laptime=90.0, tire_model=toy_tire_model, n_simulations=300, seed=21)
    assert compare_strategies(STRATEGIES, **kwargs) == compare_strategies(STRATEGIES, **kwargs)


def test_run_monte_carlo_noise_off_matches_simulate_race(toy_tire_model):
    strategy = STRATEGIES["2-stop"]
    result = run_monte_carlo(
        strategy,
        base_laptime=90.0,
        tire_model=toy_tire_model,
        n_simulations=5,
        lap_time_noise_std=0.0,
        pit_loss_noise_std=0.0,
    )
    reference = simulate_race(strategy, base_laptime=90.0, tire_model=toy_tire_model).total_time
    assert np.allclose(result.total_times, reference, rtol=0, atol=1e-9)


# --- T-007: equal-length guard + paired delta vs best-by-mean ---


def test_compare_strategies_rejects_unequal_total_laps(toy_tire_model):
    strategies = {"long": [Stint("HARD", 40)], "short": [Stint("MEDIUM", 15), Stint("HARD", 15)]}
    with pytest.raises(ValueError, match="same total number of laps"):
        compare_strategies(strategies, base_laptime=90.0, tire_model=toy_tire_model, n_simulations=10)


def test_unequal_lengths_still_allowed_in_single_strategy_and_low_level_paths(toy_tire_model):
    # run_monte_carlo (one strategy) and simulate_totals (the primitive)
    # have no equal-length constraint -- only compare_strategies does
    r = run_monte_carlo([Stint("HARD", 37)], 90.0, toy_tire_model, n_simulations=5, seed=1)
    assert r.total_times.shape == (5,)
    t = simulate_totals(
        {"a": [Stint("HARD", 20)], "b": [Stint("HARD", 30)]}, 90.0, toy_tire_model, n_simulations=5
    )
    assert set(t) == {"a", "b"}


def test_paired_delta_reference_is_best_mean_and_has_zero_delta(toy_tire_model):
    summary = compare_strategies(
        STRATEGIES, base_laptime=90.0, tire_model=toy_tire_model, n_simulations=2000, seed=13
    )
    best = min(summary, key=lambda n: summary[n]["mean"])
    for name, stats in summary.items():
        assert stats["delta_reference"] == best
        # mean of paired differences == difference of means
        assert stats["delta_mean"] == pytest.approx(stats["mean"] - summary[best]["mean"], abs=1e-9)
        assert stats["delta_p10"] <= stats["delta_p90"]
        if name != best:
            assert stats["delta_mean"] > 0
    ref = summary[best]
    assert ref["delta_mean"] == ref["delta_p10"] == ref["delta_p90"] == 0.0


def test_paired_delta_noise_off_equals_deterministic_gap(toy_tire_model):
    summary = compare_strategies(
        STRATEGIES,
        base_laptime=90.0,
        tire_model=toy_tire_model,
        n_simulations=10,
        lap_time_noise_std=0.0,
        pit_loss_noise_std=0.0,
        seed=2,
    )
    det = {n: simulate_race(s, 90.0, toy_tire_model).total_time for n, s in STRATEGIES.items()}
    best = min(det, key=det.get)
    for name, stats in summary.items():
        gap = det[name] - det[best]
        assert stats["delta_mean"] == pytest.approx(gap, abs=1e-9)
        assert stats["delta_p10"] == pytest.approx(gap, abs=1e-9)
        assert stats["delta_p90"] == pytest.approx(gap, abs=1e-9)


def test_paired_delta_is_constant_for_same_length_same_stops_under_crn(toy_tire_model):
    """
    The point of pairing: with shared draws, two 1-stop strategies of the
    same length differ by the deterministic gap in EVERY iteration, so
    delta_p10 == delta_p90 even with large noise -- while each
    strategy's own p10..p90 spread is wide.
    """
    a = [Stint("MEDIUM", 10), Stint("HARD", 10)]
    b = [Stint("SOFT", 8), Stint("HARD", 12)]
    summary = compare_strategies(
        {"a": a, "b": b},
        base_laptime=90.0,
        tire_model=toy_tire_model,
        n_simulations=1000,
        lap_time_noise_std=3.0,
        pit_loss_noise_std=5.0,
        seed=9,
    )
    det_gap = abs(
        simulate_race(a, 90.0, toy_tire_model).total_time
        - simulate_race(b, 90.0, toy_tire_model).total_time
    )
    loser = max(summary, key=lambda n: summary[n]["mean"])
    s = summary[loser]
    assert s["delta_p10"] == pytest.approx(det_gap, abs=1e-6)
    assert s["delta_p90"] == pytest.approx(det_gap, abs=1e-6)
    assert s["p90"] - s["p10"] > 10.0  # marginal spread is wide; paired spread is zero


def test_paired_delta_spread_comes_only_from_unshared_pit_draws(toy_tire_model):
    """1-stop vs 2-stop, same length: the only unshared draw is the 2nd stop's pit noise."""
    sigma_p = 5.0
    summary = compare_strategies(
        {
            "one": [Stint("MEDIUM", 15), Stint("HARD", 15)],
            "two": [Stint("MEDIUM", 10), Stint("HARD", 10), Stint("MEDIUM", 10)],
        },
        base_laptime=90.0,
        tire_model=toy_tire_model,
        n_simulations=100_000,
        lap_time_noise_std=0.8,
        pit_loss_noise_std=sigma_p,
        seed=17,
    )
    other = [n for n in summary if n != summary[n]["delta_reference"]][0]
    s = summary[other]
    # p90 - p10 of N(mu, sigma) is 2 * 1.2816 * sigma
    assert s["delta_p90"] - s["delta_p10"] == pytest.approx(2 * 1.2816 * sigma_p, rel=0.03)


def test_paired_delta_tie_goes_to_first_listed(toy_tire_model):
    same = [Stint("MEDIUM", 10), Stint("HARD", 10)]
    summary = compare_strategies(
        {"first": same, "second": list(same)}, 90.0, toy_tire_model, n_simulations=100, seed=1
    )
    assert summary["first"]["delta_reference"] == "first"
    assert summary["second"]["delta_mean"] == 0.0
