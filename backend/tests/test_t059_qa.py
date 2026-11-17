"""T-059 milestone QA: edge cases and invariants for race_grid_sim / calibration loading / chat tool loop.

Tests marked xfail document confirmed bugs (strict=True: they flip to a failure when the owner fixes the bug, so the
marker must then be removed).
"""

import json

import numpy as np
import pytest

from app.core.race_grid_sim import (
    CarSpec,
    load_grid_calibration,
    load_interaction_params,
    simulate_grid,
)
from app.core.race_simulator import Stint
from app.services import chat_service

BASE = 90.0


def field(d=6, laps=20, p_dnf=0.0):
    plan = [Stint("MEDIUM", laps // 2), Stint("HARD", laps - laps // 2)]
    return [CarSpec(f"C{i}", f"T{i // 2}", i + 1, 0.1 * i, list(plan), p_dnf) for i in range(d)]


# ---------------------------------------------------------------- calibration loader robustness (BUG)

def test_calibration_non_numeric_value_does_not_raise(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"version": 1, "sc": {"gap_s": "abc"}}))
    cal = load_grid_calibration(p)
    assert cal.sc_gap_s == 0.9


def test_calibration_json_list_does_not_raise(tmp_path):
    p = tmp_path / "c.json"
    p.write_text("[]")
    cal = load_grid_calibration(p)
    assert cal.source.startswith("placeholder")


def test_interaction_null_section_does_not_raise(tmp_path):
    p = tmp_path / "i.json"
    p.write_text(json.dumps({"version": 1, "overtake": None}))
    load_interaction_params(p)


def test_calibration_nan_and_missing_keys_fall_back(tmp_path):
    p = tmp_path / "c.json"
    p.write_text('{"version": 2, "sc": {"gap_s": NaN}}')
    cal = load_grid_calibration(p)
    assert cal.sc_gap_s == 0.9 and "sc.gap_s" in cal.placeholder_keys


# ---------------------------------------------------------------- sim edge cases / invariants

def test_single_run_two_laps_two_cars(toy_tire_model):
    cars = field(2, 2)
    for c in cars:
        c.strategy = [Stint("MEDIUM", 1), Stint("HARD", 1)]
    r = simulate_grid(cars, BASE, toy_tire_model, 2, n_sims=1, seed=1)
    assert sorted(r.positions[0].tolist()) == [1, 2]


def test_one_lap_race_runs(toy_tire_model):
    cars = field(3, 1)
    for c in cars:
        c.strategy = [Stint("MEDIUM", 1)]
    r = simulate_grid(cars, BASE, toy_tire_model, 1, n_sims=3, seed=1)
    assert r.positions.shape == (3, 3)


def test_single_car_rejected(toy_tire_model):
    with pytest.raises(ValueError):
        simulate_grid(field(1), BASE, toy_tire_model, 20, n_sims=2)


def test_positions_are_permutations_and_retired_last(toy_tire_model):
    r = simulate_grid(field(8, 24, p_dnf=0.4), BASE, toy_tire_model, 24, n_sims=300, seed=5, forced_sc_lap=8)
    for row in r.positions:
        assert sorted(row.tolist()) == list(range(1, 9))
    # every classified car finishes ahead of every retired car
    for k in range(r.n_sims):
        ret = r.retired[k]
        if ret.any() and (~ret).any():
            assert r.positions[k][~ret].max() < r.positions[k][ret].min()
    assert np.isnan(r.finish_time_s[r.retired]).all()


def test_finish_times_sorted_by_position(toy_tire_model):
    r = simulate_grid(field(8, 24), BASE, toy_tire_model, 24, n_sims=100, seed=2)
    for k in range(r.n_sims):
        o = np.argsort(r.positions[k])
        t = r.finish_time_s[k][o]
        assert (np.diff(t) >= -1e-9).all()


def test_deterministic_and_crn_unchanged_cars(toy_tire_model):
    a = simulate_grid(field(6, 20), BASE, toy_tire_model, 20, n_sims=50, seed=9)
    b = simulate_grid(field(6, 20), BASE, toy_tire_model, 20, n_sims=50, seed=9)
    assert (a.positions == b.positions).all()
    assert np.array_equal(a.finish_time_s, b.finish_time_s, equal_nan=True)
    # DNFs are CRN: changing one car's plan must not change who retires
    cars = field(6, 20, p_dnf=0.3)
    c2 = field(6, 20, p_dnf=0.3)
    c2[3].strategy = [Stint("SOFT", 8), Stint("HARD", 12)]
    x = simulate_grid(cars, BASE, toy_tire_model, 20, n_sims=200, seed=4)
    y = simulate_grid(c2, BASE, toy_tire_model, 20, n_sims=200, seed=4)
    assert (x.retired == y.retired).all()


def test_pit_rejoin_gap_is_pit_loss_without_interactions(toy_tire_model):
    """Two identical cars, car 0 stops on lap 10 and car 1 does not: car 0 must end ~pit_loss behind (no noise)."""
    s0 = [Stint("MEDIUM", 9), Stint("MEDIUM", 11)]
    s1 = [Stint("MEDIUM", 20)]
    # same compounds/ages differ after the stop; compare against a no-stop copy of itself instead
    cars = [CarSpec("A", "X", 1, 0.0, s0), CarSpec("B", "Y", 2, 0.0, s1)]
    r = simulate_grid(cars, BASE, toy_tire_model, 20, n_sims=1, seed=0, interactions=False, pace_noise=False,
                      dnfs=False, lap_time_noise_std=0.0, pit_loss_noise_std=0.0, pit_loss_s=22.0)
    cars2 = [CarSpec("A", "X", 1, 0.0, s1), CarSpec("B", "Y", 2, 0.0, s1)]
    r2 = simulate_grid(cars2, BASE, toy_tire_model, 20, n_sims=1, seed=0, interactions=False, pace_noise=False,
                       dnfs=False, lap_time_noise_std=0.0, pit_loss_noise_std=0.0, pit_loss_s=22.0)
    delta = r.finish_time_s[0, 0] - r2.finish_time_s[0, 0]
    # a fresh-tyre stint can recover some tyre wear, but the stop itself must add the full 22 s
    assert 10.0 < delta < 26.0


# ---------------------------------------------------------------- chat tool loop / error paths

def test_chat_run_tool_bad_inputs():
    assert "error" in chat_service.run_tool("nope", {})
    assert "error" in chat_service.run_tool("optimize_strategy", "not json")  # missing required arg -> error dict, no raise
    assert "error" in chat_service.run_tool("optimize_strategy", {"track_id": "zzz-not-a-track"})
    assert "error" in chat_service.run_tool("get_race_results", None)


def test_chat_loop_caps_and_survives_garbage(monkeypatch):
    calls = {"n": 0}

    def fake(messages, tools):
        calls["n"] += 1
        if tools is None:
            return {"content": "final"}
        return {"content": "", "tool_calls": [{"function": {"name": "list_tracks", "arguments": "garbage"}},
                                              {"function": {"name": "unknown_tool", "arguments": {"a": 1}}},
                                              {}]}

    monkeypatch.setattr(chat_service, "_ollama_chat", fake)
    out = chat_service.chat([{"role": "user", "content": "hi"}])
    assert out["reply"] == "final"
    assert calls["n"] == chat_service.MAX_TOOL_ROUNDS + 1
    assert len(out["tool_calls"]) == 3 * chat_service.MAX_TOOL_ROUNDS


def test_chat_empty_final_reply_has_fallback(monkeypatch):
    monkeypatch.setattr(chat_service, "_ollama_chat", lambda m, t: {"content": None})
    assert chat_service.chat([{"role": "user", "content": "hi"}])["reply"]


# ---------------------------------------------------------------- CORS env parsing

def test_cors_env_parsing(monkeypatch):
    import importlib

    import app.main as m

    monkeypatch.setenv("ALLOWED_ORIGINS", " https://a.vercel.app/ , ,https://b.com ")
    m2 = importlib.reload(m)
    try:
        assert m2.ALLOWED_ORIGINS == ["https://a.vercel.app", "https://b.com"]
    finally:
        monkeypatch.delenv("ALLOWED_ORIGINS")
        importlib.reload(m)
