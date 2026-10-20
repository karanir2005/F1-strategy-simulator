import time

import numpy as np
import pytest

from app.core.optimizer import MIN_STINT_LAPS, min_stint_laps, NoFeasiblePlan, _Cost, optimize
from app.core.race_simulator import Stint, simulate_race

CAPS = {"SOFT": 25, "MEDIUM": 35, "HARD": 45}
TOY_CAPS = {"SOFT": 20, "MEDIUM": 30, "HARD": 40}


def run(tire_model, **kw):
    args = dict(
        total_laps=56, pit_loss_s=21.3, base_laptime_s=98.4, max_stint_laps=CAPS,
        max_stops=3, n_sims=500, top_n=10, seed=1, pit_loss_noise_std=3.0,
    )
    args.update(kw)
    return optimize(tire_model, **args)


@pytest.fixture(scope="module")
def real_model():
    from app.api.routes.simulation import get_tire_model
    return get_tire_model()


def test_screen_cost_matches_simulate_race(toy_tire_model):
    cost = _Cost(toy_tire_model, list(TOY_CAPS), TOY_CAPS, 40, 90.0, 22.0)
    comps, laps = ("SOFT", "HARD", "MEDIUM"), (12, 18, 10)
    ref = simulate_race(
        [Stint(c, n) for c, n in zip(comps, laps)], 90.0, toy_tire_model, pit_loss_seconds=22.0
    )
    assert cost.total(comps, laps) == pytest.approx(ref.total_time)


def test_rules_and_caps(toy_tire_model):
    res = run(toy_tire_model, total_laps=40, max_stint_laps=TOY_CAPS)
    assert res.plans
    for p in res.plans:
        compounds = [s["compound"] for s in p["stints"]]
        assert len(set(compounds)) >= 2
        assert sum(s["laps"] for s in p["stints"]) == 40
        assert p["n_stops"] == len(p["stints"]) - 1 <= 3
        for s in p["stints"]:
            assert MIN_STINT_LAPS <= s["laps"] <= TOY_CAPS[s["compound"]]
            assert s["start_age"] == 1
        assert p["pit_laps"] == list(np.cumsum([s["laps"] for s in p["stints"]])[:-1])


def test_ranking_consistent_with_paired_deltas(toy_tire_model):
    res = run(toy_tire_model, total_laps=40, max_stint_laps=TOY_CAPS)
    means = [p["mean_s"] for p in res.plans]
    assert means == sorted(means)
    top = res.plans[0]
    assert top["delta_to_best_mean_s"] == 0.0
    assert top["delta_to_best_p10_s"] == 0.0 == top["delta_to_best_p90_s"]
    for p in res.plans:
        assert p["delta_to_best_mean_s"] == pytest.approx(p["mean_s"] - means[0], abs=1e-6)
        assert p["delta_to_best_mean_s"] >= 0
        assert p["delta_to_best_p10_s"] <= p["delta_to_best_p90_s"]
        assert p["p10_s"] <= p["p90_s"]
    assert [p["rank"] for p in res.plans] == list(range(1, len(res.plans) + 1))


def test_deterministic_with_seed(toy_tire_model):
    kw = dict(total_laps=40, max_stint_laps=TOY_CAPS)
    assert run(toy_tire_model, **kw).plans == run(toy_tire_model, **kw).plans


def test_best_plan_is_locally_optimal(toy_tire_model):
    res = run(toy_tire_model, total_laps=40, max_stint_laps=TOY_CAPS)
    best = res.plans[0]
    comps = tuple(s["compound"] for s in best["stints"])
    laps = [s["laps"] for s in best["stints"]]
    cost = _Cost(toy_tire_model, list(TOY_CAPS), TOY_CAPS, 40, 98.4, 21.3)
    base = cost.total(comps, laps)
    for i in range(len(laps)):
        for j in range(len(laps)):
            if i == j:
                continue
            cand = list(laps)
            cand[i] -= 1
            cand[j] += 1
            if cand[i] >= MIN_STINT_LAPS and cand[j] <= TOY_CAPS[comps[j]]:
                assert cost.total(comps, cand) >= base - 1e-9


def test_max_stops_limits_and_infeasible(toy_tire_model):
    res = run(toy_tire_model, total_laps=40, max_stops=1, max_stint_laps=TOY_CAPS)
    assert {p["n_stops"] for p in res.plans} == {1}
    with pytest.raises(NoFeasiblePlan):
        run(toy_tire_model, total_laps=100, max_stint_laps={"SOFT": 10, "MEDIUM": 10, "HARD": 10})
    with pytest.raises(NoFeasiblePlan):
        run(toy_tire_model, max_stint_laps={"HARD": 45})


def test_timing_real_model(real_model):
    run(real_model, n_sims=100)  # warm up
    t0 = time.perf_counter()
    res = run(real_model, n_sims=2000, top_n=5)
    elapsed = time.perf_counter() - t0
    print(f"optimize max_stops=3 n_sims=2000: {elapsed:.2f}s, screened={res.n_candidates_screened}")
    assert elapsed < 3.0
    assert res.plans[0]["n_stops"] >= 1


def test_no_reordered_duplicates(toy_tire_model):
    res = run(toy_tire_model, total_laps=40, max_stint_laps=TOY_CAPS)
    keys = [tuple(sorted((s["compound"], s["laps"]) for s in p["stints"])) for p in res.plans]
    assert len(keys) == len(set(keys))


def test_cap_floor(toy_tire_model):
    # observed caps of 10 are floored to 15, so 3 stints can cover 40 laps
    res = run(toy_tire_model, total_laps=40, max_stops=2, max_stint_laps={"SOFT": 10, "MEDIUM": 10, "HARD": 10})
    assert res.plans
    assert all(s["laps"] <= 15 for p in res.plans for s in p["stints"])
    assert any(s["laps"] > 10 for p in res.plans for s in p["stints"])


def test_min_stint_scales_with_distance():
    assert min_stint_laps(56) == 8
    assert min_stint_laps(78) == 8
    assert min_stint_laps(120) == 12
    assert min_stint_laps(10) == 5  # clamped so short distances stay feasible


def test_no_token_stints_with_tiny_caps(toy_tire_model):
    # caps below the 15-lap floor used to allow 5-lap first stints; now every stint >= min
    res = run(toy_tire_model, total_laps=44, max_stint_laps={"SOFT": 9, "MEDIUM": 12, "HARD": 12}, top_n=10)
    lo = min_stint_laps(44)
    for p in res.plans:
        assert all(s["laps"] >= lo for s in p["stints"])


def test_effective_caps_exposed(toy_tire_model):
    res = run(toy_tire_model, total_laps=40, max_stint_laps={"SOFT": 9, "MEDIUM": 30, "HARD": 100})
    assert res.effective_caps == {"SOFT": 15, "MEDIUM": 30, "HARD": 40}  # floor 15, clip to race laps
    assert res.min_stint_laps == min_stint_laps(40)


def test_real_model_min_stint_and_runtime(real_model):
    t = time.perf_counter()
    res = run(real_model, total_laps=70, max_stint_laps={"SOFT": 12, "MEDIUM": 30, "HARD": 40}, n_sims=2000)
    assert time.perf_counter() - t < 3.0
    assert all(s["laps"] >= 8 for p in res.plans for s in p["stints"])
    again = run(real_model, total_laps=70, max_stint_laps={"SOFT": 12, "MEDIUM": 30, "HARD": 40}, n_sims=2000)
    assert [p["stints"] for p in res.plans] == [p["stints"] for p in again.plans]
