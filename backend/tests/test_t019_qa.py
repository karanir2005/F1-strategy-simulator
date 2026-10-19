"""T-019 QA: independent checks of track_params.json, the optimizer and the API on REAL track data."""
import itertools
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.routes.simulation import get_tire_model
from app.core.optimizer import MIN_STINT_CAP, MIN_STINT_LAPS, optimize
from app.core.race_simulator import Stint, simulate_race
from app.core.track_params import set_track_params_path
from app.main import app

REAL = Path(__file__).resolve().parents[2] / "data" / "tracks" / "track_params.json"
DRY = ("SOFT", "MEDIUM", "HARD")

pytestmark = pytest.mark.skipif(not REAL.exists(), reason="real track_params.json missing")


@pytest.fixture(scope="module")
def tracks():
    return json.loads(REAL.read_text())


@pytest.fixture(scope="module")
def tm():
    return get_tire_model()


@pytest.fixture()
def client():
    set_track_params_path(REAL)
    yield TestClient(app)
    set_track_params_path(None)


def _run(tm, t, **kw):
    return optimize(
        tm, t["total_laps"], t["pit_loss_s"], t["base_laptime_s"], t["max_stint_laps"],
        n_sims=kw.pop("n_sims", 200), seed=0, **kw,
    )


def test_real_values_plausible(tracks):
    assert len(tracks) == 23 and len({t["track_id"] for t in tracks}) == 23
    for t in tracks:
        assert 40 <= t["total_laps"] <= 80, t["track_id"]
        assert 15 <= t["pit_loss_s"] <= 35, t["track_id"]
        assert 60 <= t["base_laptime_s"] <= 115, t["track_id"]
        assert 0 < t["pit_loss_noise_std_s"] <= 10, t["track_id"]
        assert len(t["max_stint_laps"]) >= 2, t["track_id"]
        assert all(0 < v <= t["total_laps"] for v in t["max_stint_laps"].values())


def test_all_tracks_plans_valid(tracks, tm):
    for t in tracks:
        res = _run(tm, t, top_n=10)
        assert res.plans, t["track_id"]
        for p in res.plans:
            comps = [s["compound"] for s in p["stints"]]
            laps = [s["laps"] for s in p["stints"]]
            assert sum(laps) == t["total_laps"]
            assert len(set(comps)) >= 2
            assert min(laps) >= MIN_STINT_LAPS
            assert p["n_stops"] == len(laps) - 1 <= 3
            for c, n in zip(comps, laps):
                assert n <= max(t["max_stint_laps"][c], MIN_STINT_CAP)
        keys = [tuple(sorted((s["compound"], s["laps"]) for s in p["stints"])) for p in res.plans]
        assert len(keys) == len(set(keys)), f"{t['track_id']} duplicate multiset"
        means = [p["mean_s"] for p in res.plans]
        assert means == sorted(means)


def test_determinism_real(tracks, tm):
    t = tracks[0]
    assert _run(tm, t).plans == _run(tm, t).plans


def test_closed_form_matches_simulate_race(tracks, tm):
    # n_sims=1 with zero noise would be ideal; instead check the deterministic total via simulate_race
    from app.core.optimizer import _Cost

    t = next(x for x in tracks if x["track_id"] == "austin")
    caps = {c: min(max(v, MIN_STINT_CAP), t["total_laps"]) for c, v in t["max_stint_laps"].items()}
    cost = _Cost(tm, list(caps), caps, t["total_laps"], t["base_laptime_s"], t["pit_loss_s"])
    for comps, laps in [(("MEDIUM", "SOFT"), (28, 28)), (("HARD", "MEDIUM", "SOFT"), (10, 20, 26))]:
        sim = simulate_race([Stint(c, n) for c, n in zip(comps, laps)], t["base_laptime_s"], tm, t["pit_loss_s"]).total_time
        assert cost.total(comps, laps) == pytest.approx(sim, abs=1e-6)


def test_brute_force_two_stop_best(tracks, tm):
    """Optimizer's best deterministic plan must be (near) the brute-force optimum over <=2 stops."""
    t = next(x for x in tracks if x["track_id"] == "austin")
    caps = {c: min(max(v, MIN_STINT_CAP), t["total_laps"]) for c, v in t["max_stint_laps"].items()}
    L = t["total_laps"]

    def tot(comps, laps):
        return simulate_race([Stint(c, n) for c, n in zip(comps, laps)], t["base_laptime_s"], tm, t["pit_loss_s"]).total_time

    best = (1e18, None)
    for ns in (2, 3):
        for seq in itertools.product(DRY, repeat=ns):
            if len(set(seq)) < 2:
                continue
            rng = range(MIN_STINT_LAPS, 60, 1 if ns == 2 else 2)
            for head in itertools.product(rng, repeat=ns - 1):
                last = L - sum(head)
                laps = head + (last,)
                if last < MIN_STINT_LAPS or any(n > caps[c] for c, n in zip(seq, laps)):
                    continue
                v = tot(seq, laps)
                if v < best[0]:
                    best = (v, (seq, laps))
    res = _run(tm, t, max_stops=2, n_sims=2000)
    opt_det = min(
        tot([s["compound"] for s in p["stints"]], [s["laps"] for s in p["stints"]]) for p in res.plans
    )
    assert opt_det <= best[0] + 1.0, (opt_det, best)  # within 1 s of brute-force (MC ranking may differ by noise)


def test_api_paths(client):
    tr = client.get("/tracks")
    assert tr.status_code == 200 and len(tr.json()) == 23
    assert client.post("/optimize", json={"track_id": "nope"}).status_code == 404
    assert client.post("/simulate", json={"track_id": "nope", "strategies": [{"name": "a", "stints": [{"compound": "SOFT", "laps": 10}]}]}).status_code in (404, 422)
    for bad in ({"max_stops": 4}, {"max_stops": 0}, {"n_sims": 20001}, {"top_n": 11}, {"top_n": 0}, {"total_laps": 0}):
        assert client.post("/optimize", json={"track_id": "austin", **bad}).status_code == 422, bad
    r = client.post("/optimize", json={"track_id": "austin", "n_sims": 200})
    assert r.status_code == 200
    b = r.json()
    assert set(b) == {"track", "params_used", "strategies", "n_candidates_screened", "n_candidates_simulated", "assumptions"}
    assert set(b["strategies"][0]) == {
        "rank", "n_stops", "stints", "pit_laps", "mean_s", "p10_s", "p90_s",
        "delta_to_best_mean_s", "delta_to_best_p10_s", "delta_to_best_p90_s",
    }
    set_track_params_path(Path("does/not/exist.json"))
    assert client.get("/tracks").status_code == 503
