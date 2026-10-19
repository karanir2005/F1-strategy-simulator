import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_simulate_returns_results_for_each_strategy():
    payload = {
        "base_laptime": 90.0,
        "strategies": [
            {
                "name": "1-stop",
                "stints": [
                    {"compound": "MEDIUM", "laps": 20},
                    {"compound": "HARD", "laps": 20},
                ],
            },
            {
                "name": "2-stop",
                "stints": [
                    {"compound": "SOFT", "laps": 10},
                    {"compound": "MEDIUM", "laps": 15},
                    {"compound": "HARD", "laps": 15},
                ],
            },
        ],
        "n_simulations": 100,
        "seed": 42,
    }
    resp = client.post("/simulate", json=payload)
    assert resp.status_code == 200

    body = resp.json()
    names = {r["name"] for r in body["results"]}
    assert names == {"1-stop", "2-stop"}
    for result in body["results"]:
        assert result["p10"] <= result["p50"] <= result["p90"]
        assert 0.0 <= result["win_probability"] <= 1.0
        assert len(result["lap_times"]) == 40  # 20 + 20 laps for both strategies
        assert isinstance(result["pit_laps"], list)

    one_stop = next(r for r in body["results"] if r["name"] == "1-stop")
    assert one_stop["pit_laps"] == [21]  # first lap of the second stint


def test_simulate_rejects_invalid_compound():
    payload = {
        "base_laptime": 90.0,
        "strategies": [{"name": "bad", "stints": [{"compound": "ULTRASOFT", "laps": 10}]}],
    }
    resp = client.post("/simulate", json=payload)
    assert resp.status_code == 422


def test_simulate_rejects_empty_strategy_list():
    payload = {"base_laptime": 90.0, "strategies": []}
    resp = client.post("/simulate", json=payload)
    assert resp.status_code == 422


def test_simulate_rejects_non_positive_base_laptime():
    payload = {
        "base_laptime": 0,
        "strategies": [{"name": "x", "stints": [{"compound": "MEDIUM", "laps": 10}]}],
    }
    resp = client.post("/simulate", json=payload)
    assert resp.status_code == 422


def test_simulate_rejects_duplicate_strategy_names():
    """
    compare_strategies keys its internal dict by strategy name -- without
    this check, two strategies sharing a name would silently collapse
    into one result instead of erroring, quietly dropping half the
    comparison with no indication anything was wrong.
    """
    payload = {
        "base_laptime": 90.0,
        "strategies": [
            {"name": "dup", "stints": [{"compound": "HARD", "laps": 10}]},
            {"name": "dup", "stints": [{"compound": "SOFT", "laps": 10}]},
        ],
    }
    resp = client.post("/simulate", json=payload)
    assert resp.status_code == 422


def _start_age_payload(start_age=None):
    second = {"compound": "HARD", "laps": 20}
    if start_age is not None:
        second["start_age"] = start_age
    return {
        "base_laptime": 90.0,
        "strategies": [{"name": "s", "stints": [{"compound": "MEDIUM", "laps": 20}, second]}],
        "n_simulations": 50,
        "lap_time_noise_std": 0.0,
        "pit_loss_noise_std": 0.0,
        "seed": 1,
    }


def test_simulate_accepts_start_age_and_it_slows_the_stint():
    default = client.post("/simulate", json=_start_age_payload()).json()["results"][0]
    new_set = client.post("/simulate", json=_start_age_payload(1)).json()["results"][0]
    used = client.post("/simulate", json=_start_age_payload(10)).json()["results"][0]
    assert default["mean"] == new_set["mean"]
    assert used["mean"] > new_set["mean"]
    assert used["pit_laps"] == [21]  # a stop onto a used set is still a stop
    assert used["lap_times"][20] > new_set["lap_times"][20]


def test_simulate_rejects_start_age_below_one():
    payload = {
        "base_laptime": 90.0,
        "strategies": [
            {"name": "bad", "stints": [{"compound": "SOFT", "laps": 10, "start_age": 0}]}
        ],
    }
    resp = client.post("/simulate", json=payload)
    assert resp.status_code == 422


# --- T-007: API hardening + paired delta ---

from app.api.routes.simulation import MAX_RACE_LAPS, MAX_START_AGE, MAX_STRATEGIES  # noqa: E402


def _payload(strategies, **overrides):
    payload = {"base_laptime": 90.0, "strategies": strategies, "n_simulations": 200, "seed": 3}
    payload.update(overrides)
    return payload


def _strategy(name, *stints):
    return {"name": name, "stints": [dict(s) for s in stints]}


def _error_text(resp):
    return " ".join(err["msg"] for err in resp.json()["detail"])


def test_simulate_rejects_unequal_total_laps_with_clear_message():
    resp = client.post(
        "/simulate",
        json=_payload(
            [
                _strategy("forty", {"compound": "HARD", "laps": 40}),
                _strategy("thirty", {"compound": "MEDIUM", "laps": 15}, {"compound": "HARD", "laps": 15}),
            ]
        ),
    )
    assert resp.status_code == 422
    msg = _error_text(resp)
    assert "same total race laps" in msg
    assert "'forty': 40" in msg and "'thirty': 30" in msg


def test_simulate_single_strategy_has_no_equal_length_constraint():
    resp = client.post("/simulate", json=_payload([_strategy("solo", {"compound": "HARD", "laps": 37})]))
    assert resp.status_code == 200


def test_simulate_caps_laps_per_stint():
    ok = client.post("/simulate", json=_payload([_strategy("s", {"compound": "HARD", "laps": MAX_RACE_LAPS})]))
    bad = client.post("/simulate", json=_payload([_strategy("s", {"compound": "HARD", "laps": MAX_RACE_LAPS + 1})]))
    assert ok.status_code == 200
    assert bad.status_code == 422


def test_simulate_caps_total_race_laps_across_stints():
    stints = [{"compound": "MEDIUM", "laps": 60}, {"compound": "HARD", "laps": MAX_RACE_LAPS - 59}]
    resp = client.post("/simulate", json=_payload([_strategy("s", *stints)]))
    assert resp.status_code == 422
    assert f"at most {MAX_RACE_LAPS}" in _error_text(resp)


def test_simulate_caps_start_age():
    def run(age):
        return client.post(
            "/simulate",
            json=_payload([_strategy("s", {"compound": "HARD", "laps": 10, "start_age": age})]),
        )

    assert run(MAX_START_AGE).status_code == 200
    assert run(MAX_START_AGE + 1).status_code == 422
    assert run(500).status_code == 422  # T-006 QA F-2/F-3 repro


def test_simulate_caps_number_of_strategies():
    def run(n):
        strategies = [_strategy(f"s{i}", {"compound": "HARD", "laps": 10}) for i in range(n)]
        return client.post("/simulate", json=_payload(strategies, n_simulations=10))

    assert run(MAX_STRATEGIES).status_code == 200
    assert run(MAX_STRATEGIES + 1).status_code == 422


def test_simulate_returns_paired_delta_against_best_mean():
    resp = client.post(
        "/simulate",
        json=_payload(
            [
                _strategy("1-stop", {"compound": "MEDIUM", "laps": 20}, {"compound": "HARD", "laps": 20}),
                _strategy(
                    "2-stop",
                    {"compound": "SOFT", "laps": 10},
                    {"compound": "MEDIUM", "laps": 15},
                    {"compound": "HARD", "laps": 15},
                ),
            ],
            n_simulations=2000,
        ),
    )
    assert resp.status_code == 200
    body = resp.json()
    results = {r["name"]: r for r in body["results"]}
    best = min(results, key=lambda n: results[n]["mean"])
    assert body["reference_strategy"] == best

    ref = results[best]
    assert ref["delta_mean"] == ref["delta_p10"] == ref["delta_p90"] == 0.0
    for r in results.values():
        assert r["delta_mean"] == pytest.approx(r["mean"] - ref["mean"], abs=1e-6)
        assert r["delta_p10"] <= r["delta_p90"]
        # existing fields are still present (additive change)
        for key in ("mean", "std", "p10", "p50", "p90", "win_probability", "lap_times", "pit_laps"):
            assert key in r


def test_simulate_paired_delta_noise_off_is_a_point():
    resp = client.post(
        "/simulate",
        json=_payload(
            [
                _strategy("a", {"compound": "MEDIUM", "laps": 20}, {"compound": "HARD", "laps": 20}),
                _strategy("b", {"compound": "HARD", "laps": 40}),
            ],
            lap_time_noise_std=0,
            pit_loss_noise_std=0,
        ),
    )
    body = resp.json()
    for r in body["results"]:
        assert r["delta_p10"] == pytest.approx(r["delta_mean"], abs=1e-9)
        assert r["delta_p90"] == pytest.approx(r["delta_mean"], abs=1e-9)
