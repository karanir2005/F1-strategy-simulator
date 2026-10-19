"""
QA additions for T-004 (start_age, vectorized Monte Carlo, common random
numbers). Independent of the author's tests in test_race_simulator.py /
test_monte_carlo.py / test_api_simulation.py -- these probe contract edges
and statistical properties the author's tests do not cover.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.monte_carlo import compare_strategies, run_monte_carlo, simulate_totals
from app.core.race_simulator import Stint, simulate_race
from app.main import app

client = TestClient(app)


def _payload(stints, **overrides):
    payload = {
        "base_laptime": 90.0,
        "strategies": [{"name": "s", "stints": stints}],
        "n_simulations": 20,
        "seed": 1,
    }
    payload.update(overrides)
    return payload


# --- start_age: dataclass ---------------------------------------------------


def test_three_stint_strategy_with_used_sets_charges_each_stop_exactly_once(toy_tire_model):
    strategy = [Stint("SOFT", 5), Stint("MEDIUM", 5, start_age=3), Stint("HARD", 5, start_age=6)]
    with_pit = simulate_race(strategy, 90.0, toy_tire_model, pit_loss_seconds=20.0)
    no_pit = simulate_race(strategy, 90.0, toy_tire_model, pit_loss_seconds=0.0)
    assert with_pit.pit_laps == [6, 11]
    assert with_pit.total_time - no_pit.total_time == pytest.approx(40.0)
    # and only the two out-laps differ
    diffs = [a - b for a, b in zip(with_pit.lap_times, no_pit.lap_times)]
    assert [i + 1 for i, d in enumerate(diffs) if d > 1e-9] == [6, 11]


def test_used_stint_ages_match_tail_of_a_longer_fresh_stint(toy_tire_model):
    """
    Independent oracle for the age window start_age .. start_age+laps-1:
    the tyre-only part of a start_age=6, 5-lap HARD stint (linear toy curve)
    must equal laps 6..10 of a fresh 10-lap HARD stint. Fuel is anchored
    differently in the two runs, so compare lap-to-lap increments, which
    cancel the (constant-slope) fuel term.
    """
    long_fresh = simulate_race([Stint("HARD", 10)], 90.0, toy_tire_model).lap_times
    used = simulate_race([Stint("HARD", 5, start_age=6)], 90.0, toy_tire_model).lap_times
    inc_fresh = np.diff(long_fresh[5:10])
    inc_used = np.diff(used)
    assert np.allclose(inc_fresh, inc_used)


@pytest.mark.parametrize("bad", [0, -1, -100])
def test_stint_rejects_nonpositive_start_age(bad):
    with pytest.raises(ValueError):
        Stint("MEDIUM", 5, start_age=bad)


def test_start_age_far_past_reliable_age_stays_finite_and_slower_than_new(toy_tire_model):
    used = simulate_race([Stint("SOFT", 5, start_age=500)], 90.0, toy_tire_model).lap_times
    new = simulate_race([Stint("SOFT", 5)], 90.0, toy_tire_model).lap_times
    assert all(np.isfinite(used))
    assert all(u > n for u, n in zip(used, new))


# --- start_age: API ---------------------------------------------------------


@pytest.mark.parametrize("bad", [0, -1, 1.5, None])
def test_api_rejects_invalid_start_age(bad):
    resp = client.post(
        "/simulate", json=_payload([{"compound": "SOFT", "laps": 10, "start_age": bad}])
    )
    assert resp.status_code == 422


def test_api_omitted_start_age_is_identical_to_explicit_one():
    base = [{"compound": "MEDIUM", "laps": 15}, {"compound": "HARD", "laps": 15}]
    explicit = [dict(s, start_age=1) for s in base]
    a = client.post("/simulate", json=_payload(base))
    b = client.post("/simulate", json=_payload(explicit))
    assert a.status_code == b.status_code == 200
    assert a.json() == b.json()


def test_api_used_first_stint_has_no_pit_lap_and_is_slower():
    new = client.post("/simulate", json=_payload([{"compound": "HARD", "laps": 10}], lap_time_noise_std=0, pit_loss_noise_std=0)).json()["results"][0]
    used = client.post("/simulate", json=_payload([{"compound": "HARD", "laps": 10, "start_age": 5}], lap_time_noise_std=0, pit_loss_noise_std=0)).json()["results"][0]
    assert used["pit_laps"] == [] and new["pit_laps"] == []
    assert used["mean"] > new["mean"]


# --- vectorized path: statistics -------------------------------------------


def test_noisy_totals_have_analytic_mean_and_std(toy_tire_model):
    stints = [Stint("MEDIUM", 20), Stint("HARD", 20), Stint("SOFT", 10, start_age=2)]
    lap_std, pit_std, n = 0.8, 5.0, 200_000
    det = simulate_race(stints, 90.0, toy_tire_model)
    totals = simulate_totals(
        {"a": stints}, 90.0, toy_tire_model, n_simulations=n,
        lap_time_noise_std=lap_std, pit_loss_noise_std=pit_std, seed=7,
    )["a"]
    expected_std = np.sqrt(len(det.lap_times) * lap_std**2 + len(det.pit_laps) * pit_std**2)
    assert totals.mean() == pytest.approx(det.total_time, abs=5 * expected_std / np.sqrt(n))
    assert totals.std() == pytest.approx(expected_std, rel=0.01)


def test_noise_components_isolated(toy_tire_model):
    stints = [Stint("HARD", 10), Stint("SOFT", 10)]
    n = 100_000
    lap_only = simulate_totals({"a": stints}, 90.0, toy_tire_model, n_simulations=n,
                               lap_time_noise_std=0.8, pit_loss_noise_std=0.0, seed=3)["a"]
    pit_only = simulate_totals({"a": stints}, 90.0, toy_tire_model, n_simulations=n,
                               lap_time_noise_std=0.0, pit_loss_noise_std=5.0, seed=3)["a"]
    assert lap_only.std() == pytest.approx(0.8 * np.sqrt(20), rel=0.02)
    assert pit_only.std() == pytest.approx(5.0, rel=0.02)


# --- common random numbers --------------------------------------------------


def test_crn_different_race_lengths_share_a_lap_noise_prefix(toy_tire_model):
    """
    A (20 laps) and B (30 laps), no stops: lap j has the same draw in both,
    so noise(B) - noise(A) is the sum of ONLY the 10 extra laps (variance
    10 sigma^2), not 50 sigma^2 as with independent draws.
    """
    a, b = [Stint("HARD", 20)], [Stint("HARD", 30)]
    sigma, n = 0.8, 100_000
    t = simulate_totals({"a": a, "b": b}, 90.0, toy_tire_model, n_simulations=n,
                        lap_time_noise_std=sigma, pit_loss_noise_std=0.0, seed=4)
    na = t["a"] - simulate_race(a, 90.0, toy_tire_model).total_time
    nb = t["b"] - simulate_race(b, 90.0, toy_tire_model).total_time
    assert (nb - na).std() == pytest.approx(sigma * np.sqrt(10), rel=0.02)


def test_crn_kth_pit_draw_is_shared_across_stop_counts(toy_tire_model):
    one = [Stint("MEDIUM", 15), Stint("HARD", 15)]
    two = [Stint("MEDIUM", 10), Stint("HARD", 10), Stint("SOFT", 10)]
    sigma_p, n = 5.0, 100_000
    t = simulate_totals({"one": one, "two": two}, 90.0, toy_tire_model, n_simulations=n,
                        lap_time_noise_std=0.0, pit_loss_noise_std=sigma_p, seed=5)
    n1 = t["one"] - simulate_race(one, 90.0, toy_tire_model).total_time
    n2 = t["two"] - simulate_race(two, 90.0, toy_tire_model).total_time
    # shared first stop -> difference is just the second stop's draw
    assert (n2 - n1).std() == pytest.approx(sigma_p, rel=0.02)


def test_win_probabilities_sum_to_one_and_do_not_depend_on_listing_order(toy_tire_model):
    s = {
        "one": [Stint("MEDIUM", 15), Stint("HARD", 15)],
        "two": [Stint("MEDIUM", 10), Stint("HARD", 10), Stint("SOFT", 10)],
        "none": [Stint("HARD", 30)],
    }
    kw = dict(base_laptime=90.0, tire_model=toy_tire_model, n_simulations=2000, seed=8)
    fwd = compare_strategies(s, **kw)
    rev = compare_strategies(dict(reversed(list(s.items()))), **kw)
    assert sum(v["win_probability"] for v in fwd.values()) == pytest.approx(1.0)
    for name in s:
        assert fwd[name]["win_probability"] == pytest.approx(rev[name]["win_probability"], abs=1e-12)
        assert fwd[name]["mean"] == pytest.approx(rev[name]["mean"])


# --- edge cases -------------------------------------------------------------


def test_single_simulation_and_single_strategy(toy_tire_model):
    s = compare_strategies({"x": [Stint("MEDIUM", 10)]}, 90.0, toy_tire_model, n_simulations=1, seed=1)
    assert s["x"]["win_probability"] == 1.0
    assert s["x"]["std"] == 0.0
    r = run_monte_carlo([Stint("MEDIUM", 10)], 90.0, toy_tire_model, n_simulations=1, seed=1)
    assert r.total_times.shape == (1,)


def test_run_monte_carlo_signature_and_naming_backward_compatible(toy_tire_model):
    r = run_monte_carlo([Stint("MEDIUM", 10)], 90.0, toy_tire_model, 50, 0.8, 5.0, 23.5, 1, "named")
    assert r.strategy_name == "named" and len(r.total_times) == 50
    assert r.mean == pytest.approx(r.percentile(50), abs=3.0)
