"""T-060: the fitted calibration file (data/sim_params/grid_sim_calibration.json) is loaded, falls back per key, and is
actually used by the grid sim and by grid_inputs."""

import json

import numpy as np

import app.core.race_grid_sim as rgs
from app.core import grid_inputs as G
from app.core.race_grid_sim import (
    DRIVER_PACE_SD_S,
    TEAM_PACE_SD_S,
    CarSpec,
    GridCalibration,
    load_grid_calibration,
    simulate_grid,
)
from app.core.race_simulator import Stint

BASE = 90.0


def kw(**over):
    d = dict(lap_time_noise_std=0.0, pit_loss_noise_std=0.0, pace_noise=False, dnfs=False, seed=3)
    d.update(over)
    return d


def chasers(n=6, laps=30):
    """Later grid slots are faster, so every car behind is the attacker and passes are likely."""
    return [CarSpec(f"C{i}", f"T{i}", i + 1, 0.05 * (n - i), [Stint("HARD", laps)]) for i in range(n)]


def test_loads_values_from_file_with_per_key_fallback(tmp_path):
    f = tmp_path / "cal.json"
    f.write_text(json.dumps({"version": 9, "pace": {"race_pace_scale": 1.7, "driver_shrink": 0.3}, "sc": {"gap_s": 2.0},
                             "strategy_mix": {"plan_weights": [0.5, 0.3, 0.2], "pit_jitter_laps": 4},
                             "overtake_calibration": {"intercept_shift": -0.7, "held_gap_s": 0.9}}))
    cal = load_grid_calibration(f)
    assert cal.race_pace_scale == 1.7 and cal.driver_shrink == 0.3 and cal.sc_gap_s == 2.0
    assert cal.plan_weights == (0.5, 0.3, 0.2) and cal.pit_jitter_laps == 4
    assert cal.ov_intercept_shift == -0.7 and cal.held_gap_s == 0.9
    assert cal.team_pace_sd_s == TEAM_PACE_SD_S and cal.driver_pace_sd_s == DRIVER_PACE_SD_S  # not in the file: fallback
    assert "pace.team_pace_sd_s" in cal.placeholder_keys and "sc.closure" in cal.placeholder_keys
    assert "pace.race_pace_scale" not in cal.placeholder_keys
    assert cal.source == "cal.json v9"


def test_missing_or_broken_file_falls_back_to_placeholders(tmp_path):
    cal = load_grid_calibration(tmp_path / "nope.json")
    assert cal == GridCalibration(source=cal.source, placeholder_keys=cal.placeholder_keys)
    assert cal.placeholder_keys == ["all"] and cal.source.startswith("placeholder")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    cal = load_grid_calibration(bad)
    assert cal.source.startswith("placeholder") and cal.team_pace_sd_s == TEAM_PACE_SD_S


def test_shipped_calibration_file_is_complete_and_plausible():
    cal = load_grid_calibration()
    assert cal.source.startswith("grid_sim_calibration.json"), cal.source
    assert cal.placeholder_keys == [], cal.placeholder_keys
    assert 0.2 < cal.race_pace_scale < 3.0 and 0.0 <= cal.driver_shrink <= 1.5 and 0.0 <= cal.quali_weight <= 1.0
    assert 0.0 < cal.team_pace_sd_s < 1.5 and 0.0 < cal.driver_pace_sd_s < 1.5
    assert 0.3 < cal.sc_gap_s < 3.0 and 0.0 < cal.sc_gap_closure < 1.0
    assert 0.3 < cal.lap1_spacing_s < 1.5
    assert abs(sum(cal.plan_weights) - 1.0) < 1e-6 and 0 <= cal.pit_jitter_laps <= 12
    assert cal.held_gap_s is not None and 0.3 < cal.held_gap_s < 2.0


def test_sim_default_uses_loaded_calibration(toy_tire_model, monkeypatch):
    """calibration=None goes through load_grid_calibration(): a huge lap-1 spacing means nobody is ever in the attempt window."""
    monkeypatch.setattr(rgs, "load_grid_calibration", lambda: GridCalibration(lap1_spacing_s=100.0))
    r = simulate_grid(chasers(), BASE, toy_tire_model, 30, n_sims=100, **kw())
    assert r.n_overtakes.sum() == 0
    monkeypatch.setattr(rgs, "load_grid_calibration", lambda: GridCalibration(lap1_spacing_s=0.7))
    r = simulate_grid(chasers(), BASE, toy_tire_model, 30, n_sims=100, **kw())
    assert r.n_overtakes.sum() > 0


def test_overtake_intercept_shift_and_pace_scale_are_used(toy_tire_model):
    base = dict(n_sims=200, **kw())
    never = simulate_grid(chasers(), BASE, toy_tire_model, 30, calibration=GridCalibration(ov_intercept_shift=-60.0), **base)
    normal = simulate_grid(chasers(), BASE, toy_tire_model, 30, calibration=GridCalibration(), **base)
    more = simulate_grid(chasers(), BASE, toy_tire_model, 30, calibration=GridCalibration(ov_intercept_shift=1.5), **base)
    assert never.n_overtakes.sum() == 0
    assert normal.n_overtakes.sum() > 0
    assert more.n_overtakes.sum() >= normal.n_overtakes.sum()
    # pace scale 0 removes the pace term: the faster cars stop gaining as much
    flat = simulate_grid(chasers(), BASE, toy_tire_model, 30, calibration=GridCalibration(ov_pace_scale=0.0), **base)
    assert flat.n_overtakes.sum() < normal.n_overtakes.sum()


def test_held_gap_is_used(toy_tire_model):
    """A failed attacker held far behind (5 s) leaves the attempt window; held at 0.3 s it attacks every lap."""
    cars = [CarSpec("LEAD", "A", 1, 0.0, [Stint("HARD", 40)]), CarSpec("CHAS", "B", 2, -0.05, [Stint("HARD", 40)])]
    kwargs = dict(n_sims=300, **kw())
    tight = simulate_grid(cars, BASE, toy_tire_model, 40, calibration=GridCalibration(held_gap_s=0.3, ov_intercept_shift=-3.0), **kwargs)
    loose = simulate_grid(cars, BASE, toy_tire_model, 40, calibration=GridCalibration(held_gap_s=5.0, ov_intercept_shift=-3.0), **kwargs)
    assert tight.n_overtakes.sum() > loose.n_overtakes.sum()


def test_pace_noise_and_sc_constants_are_used(toy_tire_model):
    cars = [CarSpec(f"C{i}", f"T{i // 2}", i + 1, 0.0, [Stint("HARD", 30)]) for i in range(6)]
    a = simulate_grid(cars, BASE, toy_tire_model, 30, n_sims=300, calibration=GridCalibration(team_pace_sd_s=0.0, driver_pace_sd_s=0.0),
                      **kw(pace_noise=True))
    b = simulate_grid(cars, BASE, toy_tire_model, 30, n_sims=300, calibration=GridCalibration(team_pace_sd_s=1.5, driver_pace_sd_s=1.0),
                      **kw(pace_noise=True))
    # equal cars, no noise: the lap-1 shuffle is the only source of disorder; with big pace noise the order scrambles more
    disorder = lambda r: float(np.abs(r.positions.astype(float) - r.grid[None, :]).mean())  # noqa: E731
    assert disorder(b) > disorder(a)


def test_grid_inputs_constants_come_from_the_file():
    cal = load_grid_calibration()
    assert G.RACE_PACE_SCALE == cal.race_pace_scale
    assert G.DRIVER_SHRINK == cal.driver_shrink
    assert G.QUALI_WEIGHT == cal.quali_weight
    assert tuple(G.PLAN_WEIGHTS) == tuple(cal.plan_weights)
    assert G.PIT_JITTER_LAPS == cal.pit_jitter_laps
    # explicit arguments still win (the validation passes fold-fitted values)
    rng = np.random.default_rng(0)
    plan = [Stint("MEDIUM", 20), Stint("HARD", 30)]
    same = G.vary_plan(plan, rng, 50, jitter=0)
    assert [s.laps for s in same] == [20, 30]
