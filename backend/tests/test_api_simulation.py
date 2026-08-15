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
