"""
QA additions for T-007 (API hardening + paired delta). Independent of the
author's tests in test_monte_carlo.py / test_api_simulation.py: the paired
delta is recomputed here from simulate_totals' raw per-simulation arrays,
and the API bounds are probed at their exact boundaries.
"""

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.routes.simulation import MAX_RACE_LAPS, MAX_START_AGE, MAX_STRATEGIES
from app.core import monte_carlo
from app.core.monte_carlo import compare_strategies, run_monte_carlo, simulate_totals
from app.core.race_simulator import Stint
from app.main import app

client = TestClient(app)


def _req(strategies, **overrides):
    payload = {"base_laptime": 90.0, "strategies": strategies, "n_simulations": 50, "seed": 3}
    payload.update(overrides)
    return payload


def _strat(name, *stints):
    return {"name": name, "stints": [dict(compound=c, laps=n, **kw) for c, n, kw in stints]}


def _s(compound, laps, **kw):
    return (compound, laps, kw)


# --- N-2: unequal total laps ------------------------------------------------


def test_compare_strategies_unequal_laps_raises_and_names_totals(toy_tire_model):
    strategies = {
        "long": [Stint("MEDIUM", 20), Stint("HARD", 20)],
        "short": [Stint("MEDIUM", 30)],
    }
    with pytest.raises(ValueError) as exc:
        compare_strategies(strategies, 90.0, toy_tire_model, n_simulations=10, seed=1)
    msg = str(exc.value)
    assert "long" in msg and "40" in msg and "short" in msg and "30" in msg


def test_compare_strategies_off_by_one_lap_raises(toy_tire_model):
    strategies = {"a": [Stint("HARD", 40)], "b": [Stint("HARD", 39)]}
    with pytest.raises(ValueError):
        compare_strategies(strategies, 90.0, toy_tire_model, n_simulations=10, seed=1)


def test_compare_strategies_equal_totals_with_different_stint_split_ok(toy_tire_model):
    strategies = {
        "a": [Stint("MEDIUM", 20), Stint("HARD", 20)],
        "b": [Stint("SOFT", 10), Stint("MEDIUM", 15), Stint("HARD", 15)],
        "c": [Stint("HARD", 40)],
    }
    out = compare_strategies(strategies, 90.0, toy_tire_model, n_simulations=20, seed=1)
    assert set(out) == {"a", "b", "c"}


def test_single_strategy_paths_unaffected_by_laps_check(toy_tire_model):
    res = run_monte_carlo([Stint("HARD", 7)], 90.0, toy_tire_model, n_simulations=5, seed=1)
    assert len(res.total_times) == 5
    # one-strategy compare is trivially "equal"
    out = compare_strategies({"x": [Stint("HARD", 7)]}, 90.0, toy_tire_model, n_simulations=5, seed=1)
    assert out["x"]["delta_mean"] == 0.0


def test_simulate_totals_primitive_still_accepts_unequal_laps(toy_tire_model):
    totals = simulate_totals(
        {"a": [Stint("HARD", 40)], "b": [Stint("HARD", 10)]},
        90.0,
        toy_tire_model,
        n_simulations=5,
        seed=1,
    )
    assert set(totals) == {"a", "b"}


def test_api_unequal_laps_422_names_each_total():
    resp = client.post(
        "/simulate",
        json=_req([_strat("A", _s("HARD", 40)), _strat("B", _s("MEDIUM", 15), _s("HARD", 15))]),
    )
    assert resp.status_code == 422
    msg = str(resp.json()["detail"])
    assert "'A': 40" in msg and "'B': 30" in msg


def test_api_unequal_laps_off_by_one_is_422():
    resp = client.post(
        "/simulate", json=_req([_strat("A", _s("HARD", 40)), _strat("B", _s("HARD", 39))])
    )
    assert resp.status_code == 422


def test_api_single_strategy_any_length_ok():
    resp = client.post("/simulate", json=_req([_strat("A", _s("HARD", 3))]))
    assert resp.status_code == 200


# --- N-5 / F-2: caps, exact boundaries --------------------------------------


@pytest.mark.parametrize("laps,expected", [(1, 200), (100, 200), (101, 422), (0, 422), (-1, 422), (10**9, 422)])
def test_stint_laps_boundary(laps, expected):
    resp = client.post("/simulate", json=_req([_strat("A", _s("HARD", laps))], n_simulations=5))
    assert resp.status_code == expected, resp.text[:200]


def test_stint_laps_must_be_integer():
    resp = client.post("/simulate", json=_req([_strat("A", _s("HARD", 10.5))]))
    assert resp.status_code == 422


@pytest.mark.parametrize(
    "stints,expected",
    [
        ([_s("MEDIUM", 50), _s("HARD", 50)], 200),  # total 100
        ([_s("MEDIUM", 51), _s("HARD", 50)], 422),  # total 101
        ([_s("MEDIUM", 100), _s("HARD", 1)], 422),  # each stint <= 100 but total 101
        ([_s("SOFT", 34), _s("MEDIUM", 33), _s("HARD", 33)], 200),  # 100, three stints
        ([_s("SOFT", 34), _s("MEDIUM", 34), _s("HARD", 33)], 422),  # 101
    ],
)
def test_total_laps_boundary(stints, expected):
    resp = client.post("/simulate", json=_req([_strat("A", *stints)], n_simulations=5))
    assert resp.status_code == expected, resp.text[:300]


def test_total_laps_cap_message_names_strategy():
    resp = client.post(
        "/simulate", json=_req([_strat("TooLong", _s("MEDIUM", 60), _s("HARD", 41))])
    )
    assert resp.status_code == 422
    assert "TooLong" in str(resp.json()["detail"]) and "101" in str(resp.json()["detail"])


@pytest.mark.parametrize("age,expected", [(1, 200), (25, 200), (26, 422), (0, 422), (-3, 422), (500, 422)])
def test_start_age_boundary(age, expected):
    resp = client.post(
        "/simulate", json=_req([_strat("A", _s("MEDIUM", 20, start_age=age))], n_simulations=5)
    )
    assert resp.status_code == expected, resp.text[:200]


def test_start_age_null_rejected_omitted_default_ok():
    resp = client.post(
        "/simulate",
        json=_req([{"name": "A", "stints": [{"compound": "HARD", "laps": 10, "start_age": None}]}]),
    )
    assert resp.status_code == 422
    resp = client.post("/simulate", json=_req([_strat("A", _s("HARD", 10))]))
    assert resp.status_code == 200


def test_start_age_cap_applies_to_every_stint_not_just_first():
    resp = client.post(
        "/simulate",
        json=_req([_strat("A", _s("MEDIUM", 10), _s("HARD", 10, start_age=26))], n_simulations=5),
    )
    assert resp.status_code == 422
    assert "start_age" in str(resp.json()["detail"])


def test_constants_match_documented_values():
    assert (MAX_RACE_LAPS, MAX_START_AGE, MAX_STRATEGIES) == (100, 25, 20)


@pytest.mark.parametrize("n,expected", [(1, 200), (20, 200), (21, 422)])
def test_strategies_count_boundary(n, expected):
    strategies = [_strat(f"s{i}", _s("HARD", 10)) for i in range(n)]
    resp = client.post("/simulate", json=_req(strategies, n_simulations=5))
    assert resp.status_code == expected, resp.text[:200]


def test_empty_strategies_and_empty_stints_are_422():
    assert client.post("/simulate", json=_req([])).status_code == 422
    assert client.post("/simulate", json=_req([{"name": "A", "stints": []}])).status_code == 422


def test_invalid_combinations_never_500():
    cases = [
        # unequal AND too long
        [_strat("A", _s("HARD", 100)), _strat("B", _s("HARD", 99))],
        # duplicate names AND unequal
        [_strat("A", _s("HARD", 10)), _strat("A", _s("HARD", 20))],
        # too long, unequal and bad start_age
        [_strat("A", _s("HARD", 100, start_age=30)), _strat("B", _s("HARD", 5))],
    ]
    for strategies in cases:
        resp = client.post("/simulate", json=_req(strategies, n_simulations=5))
        assert resp.status_code == 422, resp.text[:200]


def test_worst_case_allowed_request_succeeds_fast_and_bounded():
    # 50k sims x 20 strategies x 100 laps, mix of stop counts (incl. a
    # 99-stop strategy with 1-lap stints), the largest thing the API allows
    strategies = []
    for i in range(18):
        k = 1 + (i % 6)
        base, extra = divmod(100, k)
        stints = [_s(["SOFT", "MEDIUM", "HARD"][j % 3], base + (1 if j < extra else 0)) for j in range(k)]
        strategies.append(_strat(f"s{i}", *stints))
    strategies.append(_strat("many", *[_s("HARD", 1) for _ in range(100)]))
    strategies.append(_strat("none", _s("HARD", 100)))
    assert len(strategies) == MAX_STRATEGIES
    t0 = time.perf_counter()
    resp = client.post("/simulate", json=_req(strategies, n_simulations=50_000))
    elapsed = time.perf_counter() - t0
    assert resp.status_code == 200, resp.text[:300]
    assert elapsed < 10.0
    body = resp.json()
    assert len(body["results"]) == 20
    # unconstrained total_time for 99 stops must still produce finite stats
    assert all(np.isfinite(r["delta_mean"]) for r in body["results"])


# --- N-1: paired delta ------------------------------------------------------


def _paired_scenario():
    return {
        "1stop": [Stint("MEDIUM", 20), Stint("HARD", 20)],
        "2stop": [Stint("SOFT", 10), Stint("MEDIUM", 15), Stint("HARD", 15)],
        "0stop": [Stint("HARD", 40)],
    }


def test_delta_equals_percentiles_of_paired_difference_recomputed(toy_tire_model):
    kw = dict(n_simulations=4000, lap_time_noise_std=0.8, pit_loss_noise_std=5.0, seed=11)
    out = compare_strategies(_paired_scenario(), 90.0, toy_tire_model, **kw)
    raw = simulate_totals(_paired_scenario(), 90.0, toy_tire_model, **kw)
    ref = min(raw, key=lambda n: raw[n].mean())
    assert all(v["delta_reference"] == ref for v in out.values())
    for name, arr in raw.items():
        d = arr - raw[ref]
        assert out[name]["delta_mean"] == pytest.approx(d.mean(), abs=1e-9)
        assert out[name]["delta_p10"] == pytest.approx(np.percentile(d, 10), abs=1e-9)
        assert out[name]["delta_p90"] == pytest.approx(np.percentile(d, 90), abs=1e-9)


def test_delta_percentiles_are_not_differences_of_marginal_percentiles(toy_tire_model):
    # 2stop has one extra (unshared) pit draw. Paired delta p10..p90 spread
    # = 2*1.2816*sigma_pit; the marginals' spread is much wider (lap noise
    # + pit1 + pit2). Prove the implementation matches the former, and that
    # the former differs from p10_a - p10_b.
    strategies = {"1stop": [Stint("MEDIUM", 20), Stint("HARD", 20)],
                  "2stop": [Stint("MEDIUM", 14), Stint("HARD", 13), Stint("HARD", 13)]}
    out = compare_strategies(
        strategies, 90.0, toy_tire_model, n_simulations=20000,
        lap_time_noise_std=0.8, pit_loss_noise_std=5.0, seed=5,
    )
    ref = out["1stop"]["delta_reference"]
    other = "2stop" if ref == "1stop" else "1stop"
    spread = out[other]["delta_p90"] - out[other]["delta_p10"]
    assert spread == pytest.approx(2 * 1.2816 * 5.0, rel=0.05)
    diff_of_marginals = out[other]["p10"] - out[ref]["p10"]
    assert abs(diff_of_marginals - out[other]["delta_p10"]) > 1.0


def test_delta_sign_convention_positive_means_slower(toy_tire_model):
    # HARD is deterministically slower than MEDIUM over a short race at
    # zero degradation difference? use offsets: SOFT -0.5, MEDIUM -0.2, HARD 0
    strategies = {"soft": [Stint("SOFT", 5)], "hard": [Stint("HARD", 5)]}
    out = compare_strategies(
        strategies, 90.0, toy_tire_model, n_simulations=10,
        lap_time_noise_std=0.0, pit_loss_noise_std=0.0, seed=1,
    )
    assert out["soft"]["delta_reference"] == "soft"
    assert out["soft"]["delta_mean"] == 0.0
    assert out["hard"]["delta_mean"] > 0
    gap = out["hard"]["mean"] - out["soft"]["mean"]
    assert out["hard"]["delta_mean"] == pytest.approx(gap)
    assert out["hard"]["delta_p10"] == pytest.approx(gap)
    assert out["hard"]["delta_p90"] == pytest.approx(gap)


def test_reference_deltas_exactly_zero_and_nonnegative_means_property(toy_tire_model):
    rng = np.random.default_rng(0)
    for trial in range(60):
        total = int(rng.integers(5, 60))
        strategies = {}
        for k in range(int(rng.integers(1, 6))):
            n_stints = int(rng.integers(1, 4))
            cuts = sorted(rng.integers(1, total, size=n_stints - 1).tolist()) if n_stints > 1 else []
            bounds = [0] + cuts + [total]
            stints = [
                Stint(str(rng.choice(["SOFT", "MEDIUM", "HARD"])), bounds[i + 1] - bounds[i],
                      int(rng.integers(1, 10)))
                for i in range(n_stints) if bounds[i + 1] > bounds[i]
            ]
            strategies[f"s{k}"] = stints
        out = compare_strategies(
            strategies, 90.0, toy_tire_model, n_simulations=int(rng.integers(1, 200)),
            lap_time_noise_std=float(rng.choice([0.0, 0.8])),
            pit_loss_noise_std=float(rng.choice([0.0, 5.0])), seed=trial,
        )
        ref = next(iter(out.values()))["delta_reference"]
        r = out[ref]
        assert r["delta_mean"] == 0.0 and r["delta_p10"] == 0.0 and r["delta_p90"] == 0.0
        means = {n: v["mean"] for n, v in out.items()}
        assert means[ref] == min(means.values())
        for v in out.values():
            assert v["delta_p10"] <= v["delta_p90"]
            # tiny negative rounding noise is tolerable only at ~1e-9
            assert v["delta_mean"] >= -1e-9


def test_reference_is_best_mean_not_first_and_not_dependent_on_order(toy_tire_model):
    s = _paired_scenario()
    kw = dict(n_simulations=500, seed=2)
    refs = set()
    import itertools

    for perm in itertools.permutations(s):
        out = compare_strategies({n: s[n] for n in perm}, 90.0, toy_tire_model, **kw)
        refs.add(next(iter(out.values()))["delta_reference"])
    assert len(refs) == 1  # deterministic regardless of listing order


def test_reference_is_deterministic_for_same_seed(toy_tire_model):
    a = compare_strategies(_paired_scenario(), 90.0, toy_tire_model, n_simulations=300, seed=9)
    b = compare_strategies(_paired_scenario(), 90.0, toy_tire_model, n_simulations=300, seed=9)
    assert a == b


def test_identical_strategies_tie_goes_to_first_listed_for_reference_and_wins(toy_tire_model):
    strategies = {"first": [Stint("HARD", 20)], "second": [Stint("HARD", 20)]}
    out = compare_strategies(strategies, 90.0, toy_tire_model, n_simulations=100, seed=1)
    assert out["first"]["delta_reference"] == "first"
    assert out["second"]["delta_mean"] == 0.0 and out["second"]["delta_p10"] == 0.0
    assert out["first"]["win_probability"] == 1.0 and out["second"]["win_probability"] == 0.0


def test_n_simulations_one_gives_degenerate_but_valid_delta(toy_tire_model):
    out = compare_strategies(_paired_scenario(), 90.0, toy_tire_model, n_simulations=1, seed=1)
    for v in out.values():
        assert v["delta_p10"] == v["delta_p90"] == pytest.approx(v["delta_mean"])


def test_same_stop_strategies_have_constant_delta_under_heavy_noise(toy_tire_model):
    strategies = {"m": [Stint("MEDIUM", 20), Stint("HARD", 20)],
                  "s": [Stint("SOFT", 20), Stint("HARD", 20)]}
    out = compare_strategies(
        strategies, 90.0, toy_tire_model, n_simulations=2000,
        lap_time_noise_std=5.0, pit_loss_noise_std=20.0, seed=4,
    )
    for v in out.values():
        assert v["delta_p90"] - v["delta_p10"] == pytest.approx(0.0, abs=1e-6)


def test_delta_with_win_probability_inconsistency_example(toy_tire_model):
    # the N-1 motivating case: tiny gap => win_probability 0/1 but delta says "tiny"
    strategies = {"a": [Stint("MEDIUM", 20), Stint("HARD", 20)],
                  "b": [Stint("MEDIUM", 21), Stint("HARD", 19)]}
    out = compare_strategies(strategies, 90.0, toy_tire_model, n_simulations=500, seed=1)
    probs = sorted(v["win_probability"] for v in out.values())
    assert probs == [0.0, 1.0]
    loser = max(out, key=lambda n: out[n]["delta_mean"])
    assert out[loser]["delta_mean"] < 2.0


# --- API response contract --------------------------------------------------


OLD_RESULT_FIELDS = {"name", "mean", "std", "p10", "p50", "p90", "win_probability", "lap_times", "pit_laps"}


def test_api_response_is_additive_superset_of_old_fields():
    resp = client.post(
        "/simulate",
        json=_req([_strat("A", _s("MEDIUM", 20), _s("HARD", 20)),
                   _strat("B", _s("SOFT", 10), _s("MEDIUM", 15), _s("HARD", 15))]),
    )
    assert resp.status_code == 200
    body = resp.json()
    # T-030 added the optional safety_cars / assumptions top-level fields
    assert set(body) == {"reference_strategy", "results", "safety_cars", "assumptions"}
    for r in body["results"]:
        assert OLD_RESULT_FIELDS <= set(r)
        assert set(r) - OLD_RESULT_FIELDS == {"delta_mean", "delta_p10", "delta_p90"}
    names = [r["name"] for r in body["results"]]
    assert names == ["A", "B"]  # request order preserved
    ref = body["reference_strategy"]
    assert ref in names
    refrow = next(r for r in body["results"] if r["name"] == ref)
    assert refrow["delta_mean"] == refrow["delta_p10"] == refrow["delta_p90"] == 0.0
    assert refrow["mean"] == min(r["mean"] for r in body["results"])
    for r in body["results"]:
        assert r["delta_mean"] == pytest.approx(r["mean"] - refrow["mean"], abs=1e-6)


def test_api_old_style_single_strategy_request_still_works_and_reference_is_itself():
    resp = client.post("/simulate", json={
        "base_laptime": 90.0,
        "strategies": [{"name": "only", "stints": [{"compound": "HARD", "laps": 30}]}],
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["reference_strategy"] == "only"
    assert body["results"][0]["delta_mean"] == 0.0


def test_api_reference_tie_goes_to_first_listed():
    resp = client.post("/simulate", json=_req([
        _strat("first", _s("HARD", 20)), _strat("second", _s("HARD", 20))]))
    assert resp.json()["reference_strategy"] == "first"


def test_api_same_seed_same_response():
    payload = _req([_strat("A", _s("MEDIUM", 20), _s("HARD", 20)),
                    _strat("B", _s("HARD", 40))], n_simulations=300, seed=77)
    assert client.post("/simulate", json=payload).json() == client.post("/simulate", json=payload).json()


# --- N-4: caveats -----------------------------------------------------------


def test_caveats_present_in_module_and_function_docstrings():
    mod = monte_carlo.__doc__.lower()
    fn = compare_strategies.__doc__.lower()
    for doc in (mod, fn):
        assert "one car in" in doc and "isolation" in doc
        assert "additive" in doc
        assert "common random numbers" in doc
        assert "delta" in doc
    assert "same total" in mod or "total race laps" in mod
    assert "valueerror" in fn.replace(" ", "")


def test_caveats_present_in_openapi_descriptions():
    schema = client.get("/openapi.json").json()["components"]["schemas"]
    res = schema["StrategyResult"]["properties"]
    for field in ("win_probability", "delta_mean", "delta_p10", "delta_p90"):
        desc = res[field]["description"].lower()
        assert "one car in isolation" in desc, field
        assert "additive" in desc, field
        assert "common random numbers" in desc, field
    assert "lowest mean" in schema["SimulateResponse"]["properties"]["reference_strategy"]["description"].lower()
    assert "not a race-win chance" in res["win_probability"]["description"].lower()
    assert schema["StintIn"]["properties"]["start_age"]["maximum"] == 25
    assert schema["StintIn"]["properties"]["laps"]["maximum"] == 100
    assert schema["SimulateRequest"]["properties"]["strategies"]["maxItems"] == 20
