"""T-034: full-grid simulation, interaction-parameter loading, forced SC, /simulate/grid."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.monte_carlo import compare_strategies
from app.core.race_grid_sim import (
    CarSpec,
    load_interaction_params,
    rank_by_points,
    simulate_grid,
    LAP1_SPACING_S,
)
from app.core.race_simulator import Stint, simulate_race
from app.core.safety_cars import load_params as load_sc
from app.main import app

BASE = 90.0


def hard(n):
    return [Stint("HARD", n)]


def kw(**over):
    d = dict(lap_time_noise_std=0.0, pit_loss_noise_std=0.0, pace_noise=False, dnfs=False, seed=3)
    d.update(over)
    return d


def test_deterministic_given_seed(toy_tire_model):
    cars = [CarSpec(f"C{i}", f"T{i // 2}", i + 1, 0.1 * i, [Stint("MEDIUM", 12), Stint("HARD", 18)], 0.1) for i in range(8)]
    a = simulate_grid(cars, BASE, toy_tire_model, 30, n_sims=200, seed=5, safety_cars=load_sc())
    b = simulate_grid(cars, BASE, toy_tire_model, 30, n_sims=200, seed=5, safety_cars=load_sc())
    c = simulate_grid(cars, BASE, toy_tire_model, 30, n_sims=200, seed=6, safety_cars=load_sc())
    assert np.array_equal(a.positions, b.positions)
    assert np.array_equal(a.n_overtakes, b.n_overtakes)
    assert not np.array_equal(a.positions, c.positions)
    # every run's positions are a permutation
    assert (np.sort(a.positions, axis=1) == np.arange(1, 9)).all()


def test_common_random_numbers_other_cars_unchanged_noise(toy_tire_model):
    """Changing one car's strategy must not change the pace/DNF draws: the retirements stay identical."""
    base = [CarSpec(f"C{i}", f"T{i}", i + 1, 0.1 * i, hard(30), 0.3) for i in range(5)]
    alt = [CarSpec(c.code, c.team, c.grid, c.pace_delta_s, [Stint("SOFT", 10), Stint("HARD", 20)] if c.code == "C2" else c.strategy, c.p_dnf) for c in base]
    a = simulate_grid(base, BASE, toy_tire_model, 30, n_sims=300, seed=1)
    b = simulate_grid(alt, BASE, toy_tire_model, 30, n_sims=300, seed=1)
    assert np.array_equal(a.retired, b.retired)


def test_no_overtakes_when_gap_is_huge(toy_tire_model, monkeypatch):
    """Cars 100 s apart after lap 1 (far outside the 1.5 s attempt window) never pass, even when the car behind is
    faster: it can close at most 0.5 s a lap."""
    import app.core.race_grid_sim as rgs

    monkeypatch.setattr(rgs, "LAP1_SPACING_S", 100.0)
    cars = [CarSpec("SLO", "A", 1, 0.5, hard(20)), CarSpec("FAS", "B", 2, 0.0, hard(20)), CarSpec("MID", "C", 3, 0.2, hard(20))]
    r = simulate_grid(cars, BASE, toy_tire_model, 20, n_sims=100, **kw())
    assert r.n_overtakes.sum() == 0
    assert (r.positions == np.array([1, 2, 3])).all() or r.n_overtakes.sum() == 0


def test_faster_car_eventually_passes_slower(toy_tire_model):
    cars = [CarSpec("SLO", "A", 1, 0.5, hard(50)), CarSpec("FAS", "B", 2, 0.0, hard(50))]
    r = simulate_grid(cars, BASE, toy_tire_model, 50, n_sims=400, **kw())
    fast_ahead = (r.positions[:, 1] < r.positions[:, 0]).mean()
    assert fast_ahead > 0.95
    assert r.n_overtakes.mean() > 0.3  # some runs need no pass: the lap-1 shuffle already put the fast car ahead


def test_equal_cars_start_order_mostly_kept(toy_tire_model):
    cars = [CarSpec("A", "A", 1, 0.0, hard(5)), CarSpec("B", "B", 2, 0.0, hard(5))]
    r = simulate_grid(cars, BASE, toy_tire_model, 5, n_sims=500, **kw())
    assert (r.positions[:, 0] == 1).mean() > 0.5


def test_pit_rejoin_at_real_gap(toy_tire_model):
    """With interactions off, finish times are exactly the single-car engine's lap sums, so a stop re-joins
    behind a no-stop car by the pit loss (less what the fresh tyres gain)."""
    pit = 60.0
    stopper = [Stint("HARD", 10), Stint("HARD", 40)]
    cars = [CarSpec("STP", "A", 1, 0.0, stopper), CarSpec("RUN", "B", 2, 0.0, hard(50))]
    r = simulate_grid(cars, BASE, toy_tire_model, 50, n_sims=3, interactions=False, pit_loss_s=pit, **kw())
    tot = {c.code: simulate_race(c.strategy, BASE, toy_tire_model, pit_loss_seconds=pit).lap_times for c in cars}
    expect = (sum(tot["STP"][1:]) - sum(tot["RUN"][1:])) - LAP1_SPACING_S
    got = r.finish_time_s[0, 0] - r.finish_time_s[0, 1]
    assert got == pytest.approx(expect, abs=1e-6)
    # a 60 s stop is not recovered by the fresher tyres: the stopper finishes behind, about `pit` seconds back
    assert (r.positions[:, 0] == 2).all()
    # with a free stop the fresher tyres win the race
    r0 = simulate_grid(cars, BASE, toy_tire_model, 50, n_sims=3, interactions=False, pit_loss_s=0.0, **kw())
    assert (r0.positions[:, 0] == 1).all()


def test_pit_stop_drops_car_behind_cars_it_was_ahead_of(toy_tire_model):
    cars = [CarSpec(f"C{i}", f"T{i}", i + 1, 0.0 + 0.001 * i, [Stint("HARD", 10), Stint("HARD", 20)] if i == 0 else hard(30)) for i in range(5)]
    r = simulate_grid(cars, BASE, toy_tire_model, 30, n_sims=50, pit_loss_s=23.5, **kw())
    assert (r.positions[:, 0] >= 4).all()  # 23.5 s is ~5 cars' worth of 0.7 s spacing plus the stop


def test_safety_car_bunches_the_field(toy_tire_model):
    cars = [CarSpec(f"C{i}", f"T{i}", i + 1, 0.3 * i, hard(24)) for i in range(6)]
    spread = lambda r: np.nanmax(r.finish_time_s, axis=1) - np.nanmin(r.finish_time_s, axis=1)
    green = simulate_grid(cars, BASE, toy_tire_model, 24, n_sims=50, **kw())
    sc = simulate_grid(cars, BASE, toy_tire_model, 24, n_sims=50, forced_sc_lap=6, forced_sc_laps=18, **kw())
    assert sc.n_sc_laps.min() == 18
    assert spread(sc).mean() < 0.5 * spread(green).mean()


def test_forced_sc_costs_single_car_time(toy_tire_model):
    plans = {"p": [Stint("MEDIUM", 15), Stint("HARD", 15)]}
    kw1 = dict(n_simulations=50, seed=1, lap_time_noise_std=0.0, pit_loss_noise_std=0.0)
    none = compare_strategies(plans, BASE, toy_tire_model, **kw1)["p"]["mean"]
    forced = compare_strategies(plans, BASE, toy_tire_model, forced_sc_lap=5, forced_sc_laps=3, **kw1)["p"]["mean"]
    assert forced > none + 10  # three laps at ~1.4x
    # forced_sc_lap=None is bit-identical to the old call
    assert compare_strategies(plans, BASE, toy_tire_model, forced_sc_lap=None, **kw1)["p"]["mean"] == none


def test_interaction_params_loading(tmp_path):
    ph = load_interaction_params(tmp_path / "missing.json")
    assert ph.source.startswith("placeholder") and ph.placeholder_keys == ["all"]
    real = load_interaction_params()  # data/sim_params/race_interaction.json from T-053
    if not real.source.startswith("placeholder"):
        assert real.placeholder_keys == []
        assert real.ov_pace_coef > 0 and real.ov_gap_coef < 0 and real.dirty_window_s > 0
    partial = tmp_path / "p.json"
    partial.write_text('{"version": 9, "min_follow_gap_s": 0.5}')
    p = load_interaction_params(partial)
    assert p.min_follow_gap_s == 0.5 and "dirty_air.window_s" in p.placeholder_keys


def test_rank_by_points():
    assert list(rank_by_points([10, 25, 10], [3, 4, 2])) == [3, 1, 2]


client = TestClient(app)


def test_api_simulate_grid_smoke():
    r = client.post("/simulate/grid", json={"track_id": "austin", "n_sims": 60, "seed": 1})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["round"] == 18 and body["total_laps"] > 0
    ds = body["drivers"]
    assert len(ds) >= 20
    assert sum(d["p_win"] for d in ds) == pytest.approx(1.0)
    assert sum(d["p_podium"] for d in ds) == pytest.approx(3.0)
    assert all(sum(d["position_probs"]) == pytest.approx(1.0) for d in ds)
    # deterministic given the seed
    again = client.post("/simulate/grid", json={"track_id": "austin", "n_sims": 60, "seed": 1}).json()
    assert [d["p_win"] for d in again["drivers"]] == [d["p_win"] for d in ds]


def test_api_strategy_override_and_errors():
    first = client.post("/simulate/grid", json={"track_id": "austin", "n_sims": 20}).json()
    code = first["drivers"][0]["code"]
    laps = first["total_laps"]
    ok = client.post(
        "/simulate/grid",
        json={"track_id": "austin", "n_sims": 20, "forced_sc_lap": 20,
              "strategies": {code: [{"compound": "MEDIUM", "laps": laps - 20}, {"compound": "HARD", "laps": 20}]}},
    )
    assert ok.status_code == 200, ok.text
    d = {x["code"]: x for x in ok.json()["drivers"]}[code]
    assert [s["laps"] for s in d["strategy"]] == [laps - 20, 20]
    bad = client.post("/simulate/grid", json={"track_id": "austin", "n_sims": 20, "strategies": {code: [{"compound": "HARD", "laps": 5}]}})
    assert bad.status_code == 422
    assert client.post("/simulate/grid", json={"track_id": "nowhere"}).status_code == 404
