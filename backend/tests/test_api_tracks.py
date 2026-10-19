import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.race_simulator import DEFAULT_PIT_LOSS_SECONDS
from app.core.track_params import set_track_params_path
from app.main import app

FIXTURE = Path(__file__).parent / "fixtures" / "track_params_fixture.json"
client = TestClient(app)


@pytest.fixture(autouse=True)
def fixture_tracks():
    set_track_params_path(FIXTURE)
    yield
    set_track_params_path(None)


def test_get_tracks():
    resp = client.get("/tracks")
    assert resp.status_code == 200
    data = resp.json()
    assert [t["track_id"] for t in data] == ["testring", "shortring"]
    assert data[0]["total_laps"] == 56 and data[0]["max_stint_laps"]["SOFT"] == 25


def test_tracks_503_when_missing(tmp_path):
    set_track_params_path(tmp_path / "nope.json")
    for resp in (client.get("/tracks"), client.post("/optimize", json={"track_id": "x"})):
        assert resp.status_code == 503
        assert "build_track_params" in resp.json()["detail"]


def test_optimize_shape_and_defaults():
    resp = client.post("/optimize", json={"track_id": "testring", "n_sims": 300, "seed": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["params_used"] == {"total_laps": 56, "pit_loss_s": 21.3, "base_laptime_s": 98.4, "n_sims": 300}
    assert body["track"]["track_id"] == "testring"
    assert len(body["strategies"]) == 5
    best = body["strategies"][0]
    assert best["rank"] == 1 and best["delta_to_best_mean_s"] == 0
    caps = body["track"]["max_stint_laps"]
    for s in body["strategies"]:
        assert sum(x["laps"] for x in s["stints"]) == 56
        assert len({x["compound"] for x in s["stints"]}) >= 2
        assert all(x["laps"] <= caps[x["compound"]] for x in s["stints"])
    assert body["n_candidates_screened"] > body["n_candidates_simulated"] > 0
    assert body["assumptions"]


def test_optimize_overrides_and_determinism():
    req = {
        "track_id": "testring", "n_sims": 300, "seed": 5, "total_laps": 40,
        "pit_loss_s": 30.0, "base_laptime_s": 90.0, "top_n": 3,
    }
    a = client.post("/optimize", json=req).json()
    b = client.post("/optimize", json=req).json()
    assert a == b
    assert a["params_used"] == {"total_laps": 40, "pit_loss_s": 30.0, "base_laptime_s": 90.0, "n_sims": 300}
    assert len(a["strategies"]) == 3


def test_optimize_errors():
    assert client.post("/optimize", json={"track_id": "nowhere"}).status_code == 404
    bad = [
        {"max_stops": 0}, {"max_stops": 4}, {"n_sims": 20001}, {"n_sims": 0}, {"top_n": 0},
        {"top_n": 11}, {"total_laps": 0}, {"total_laps": 101}, {"pit_loss_s": -1}, {"base_laptime_s": 0},
    ]
    for extra in bad:
        assert client.post("/optimize", json={"track_id": "testring", **extra}).status_code == 422, extra
    assert client.post("/optimize", json={}).status_code == 422
    # caps cannot cover the distance -> clear 422, not a crash
    resp = client.post("/optimize", json={"track_id": "shortring", "total_laps": 100, "max_stops": 1})
    assert resp.status_code == 422


def test_optimize_timing_via_api():
    client.post("/optimize", json={"track_id": "testring", "n_sims": 100})
    t0 = time.perf_counter()
    resp = client.post("/optimize", json={"track_id": "testring", "max_stops": 3, "n_sims": 2000})
    assert resp.status_code == 200
    assert time.perf_counter() - t0 < 3.0


SIM = {
    "strategies": [
        {"name": "a", "stints": [{"compound": "MEDIUM", "laps": 20}, {"compound": "HARD", "laps": 20}]}
    ],
    "n_simulations": 200,
    "seed": 1,
}


def test_simulate_with_track_fills_defaults():
    with_track = client.post("/simulate", json={**SIM, "track_id": "testring"})
    explicit = client.post(
        "/simulate",
        json={**SIM, "base_laptime": 98.4, "pit_loss_seconds": 21.3, "pit_loss_noise_std": 3.1},
    )
    assert with_track.status_code == 200
    assert with_track.json() == explicit.json()


def test_simulate_explicit_overrides_track():
    r = client.post("/simulate", json={**SIM, "track_id": "testring", "base_laptime": 90.0, "pit_loss_seconds": 10.0})
    e = client.post(
        "/simulate",
        json={**SIM, "base_laptime": 90.0, "pit_loss_seconds": 10.0, "pit_loss_noise_std": 3.1},
    )
    assert r.json() == e.json()


def test_simulate_track_errors_and_legacy():
    assert client.post("/simulate", json={**SIM, "track_id": "nowhere"}).status_code == 404
    assert client.post("/simulate", json=SIM).status_code == 422  # no base_laptime, no track
    legacy = client.post("/simulate", json={**SIM, "base_laptime": 90.0})
    assert legacy.status_code == 200
    same = client.post(
        "/simulate", json={**SIM, "base_laptime": 90.0, "pit_loss_seconds": DEFAULT_PIT_LOSS_SECONDS}
    )
    assert legacy.json() == same.json()
