"""T-030: use_track_wear flag on /optimize, and the optional SC/VSC model."""

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.api.routes.tracks as tracks_route
from app.core.monte_carlo import simulate_totals
from app.core.race_simulator import Stint
from app.core.safety_cars import draw_timelines, load_params
from app.core.track_params import set_track_params_path
from app.main import app

FIXTURE = Path(__file__).parent / "fixtures" / "track_params_fixture.json"
client = TestClient(app)


@pytest.fixture(autouse=True)
def fixture_tracks():
    set_track_params_path(FIXTURE)
    yield
    set_track_params_path(None)


def _post(**kw):
    body = {"track_id": "testring", "n_sims": 300, "seed": 3}
    body.update(kw)
    resp = client.post("/optimize", json=body)
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_use_track_wear_default_false_and_applied_flag(monkeypatch):
    base = _post()
    assert base["params_used"]["use_track_wear"] is False
    assert base["params_used"]["safety_cars"] is False
    # no fit for the fixture track -> flag falls back to global and says so
    assert _post(use_track_wear=True)["params_used"]["use_track_wear"] is False

    model = tracks_route.get_tire_model()
    wear = {c: {"age_coef": model[c]["age_coef"] * 3} for c in ("SOFT", "MEDIUM", "HARD") if c in model}
    patched = dict(model, _track_wear={"testring": wear})
    monkeypatch.setattr(tracks_route, "get_tire_model", lambda: patched)
    off = _post()
    on = _post(use_track_wear=True)
    assert on["params_used"]["use_track_wear"] is True
    assert off["params_used"]["use_track_wear"] is False
    assert on["strategies"][0]["mean_s"] > off["strategies"][0]["mean_s"]  # 3x wear is slower


def test_safety_cars_off_is_bit_identical_and_on_is_flagged():
    off = _post()
    assert off == _post(safety_cars=False)
    assert any(a.startswith("No safety cars") for a in off["assumptions"])
    on = _post(safety_cars=True)
    assert on["params_used"]["safety_cars"] is True
    assert not any(a.startswith("No safety cars") for a in on["assumptions"])
    assert any(a.startswith("Safety cars ON") for a in on["assumptions"])
    assert on["strategies"][0]["mean_s"] > off["strategies"][0]["mean_s"]


def test_simulate_safety_cars_flag():
    body = {
        "track_id": "testring",
        "n_simulations": 500,
        "seed": 1,
        "strategies": [{"name": "a", "stints": [{"compound": "MEDIUM", "laps": 28}, {"compound": "HARD", "laps": 28}]}],
    }
    off = client.post("/simulate", json=body).json()
    on = client.post("/simulate", json=dict(body, safety_cars=True)).json()
    assert off["safety_cars"] is False and on["safety_cars"] is True
    assert on["results"][0]["mean"] > off["results"][0]["mean"]
    assert on["results"][0]["lap_times"] == off["results"][0]["lap_times"]


def test_timeline_is_shared_and_deterministic(toy_tire_model):
    p = load_params()
    strategies = {
        "a": [Stint("MEDIUM", 20), Stint("HARD", 20)],
        "b": [Stint("SOFT", 10), Stint("HARD", 30)],
    }
    kw = dict(base_laptime=90.0, tire_model=toy_tire_model, n_simulations=400, seed=5)
    r1 = simulate_totals(strategies, safety_cars=p, **kw)
    r2 = simulate_totals(strategies, safety_cars=p, **kw)
    base = simulate_totals(strategies, **kw)
    for k in strategies:
        assert np.array_equal(r1[k], r2[k])
        assert (r1[k] >= base[k] - 1e-6).any()
    # SC/VSC laps slow the race down on average
    assert r1["a"].mean() > base["a"].mean()


def test_draw_timelines_rates():
    p = load_params()
    sc, vsc = draw_timelines(np.random.default_rng(0), 20000, 60, p)
    assert not (sc & vsc).any()
    # P(at least one SC) in the data is ~0.5; the hazard model should be in that ballpark
    assert 0.35 < sc.any(axis=1).mean() < 0.65
    assert 0.3 < vsc.any(axis=1).mean() < 0.65
