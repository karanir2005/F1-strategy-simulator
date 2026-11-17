"""T-055: safety-car-reactive pit policy (pit_policy.py), the grid sim's plan_hook, POST /strategy/policy, tools."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core import pit_policy as PP
from app.core.race_grid_sim import CarSpec, simulate_grid
from app.core.race_simulator import Stint
from app.core.safety_cars import load_params as load_sc
from app.main import app

BASE = 90.0
NL = 30
CAPS = {"SOFT": 30, "MEDIUM": 30, "HARD": 30}


def make_ctx(toy, safety_cars=True):
    plan = [Stint("MEDIUM", 12), Stint("HARD", 18)]
    cars = [CarSpec(f"C{i}", f"T{i // 2}", i + 1, 0.15 * i, list(plan), 0.05) for i in range(8)]
    return PP.PolicyContext(
        cars=cars, hero=2, base_laptime_s=BASE, tire_model=toy, total_laps=NL, pit_loss_s=22.0, caps=CAPS,
        safety_cars=load_sc() if safety_cars else None,
    )


def test_apply_policy_to_plan_pull_forward_and_rules():
    plan = [Stint("MEDIUM", 12), Stint("HARD", 18)]
    new = PP.apply_policy_to_plan(plan, 8, "SOFT", CAPS)
    # the stop planned for lap 12 is pulled to lap 8; the new SOFT stint runs to where the HARD stint ended
    assert [(s.compound, s.laps) for s in new] == [("MEDIUM", 7), ("SOFT", 23)]
    late = PP.apply_policy_to_plan(plan, 20, "HARD", CAPS)  # last stint: an extra stop
    assert [(s.compound, s.laps) for s in late] == [("MEDIUM", 12), ("HARD", 7), ("HARD", 11)]
    assert sum(s.laps for s in new) == sum(s.laps for s in late) == NL
    assert PP.apply_policy_to_plan(plan, 8, "SOFT", {"SOFT": 10}) is None  # new stint longer than the cap
    assert PP.apply_policy_to_plan(plan, 1, "SOFT", CAPS) is None  # lap 1
    assert PP.apply_policy_to_plan(plan, 13, "HARD", CAPS) is None  # the plan's own stop lap
    assert PP.apply_policy_to_plan(plan, 28, "SOFT", CAPS) is None  # too few laps left
    one = [Stint("MEDIUM", 30)]
    assert PP.apply_policy_to_plan(one, 10, "MEDIUM", CAPS) is None  # would leave a single compound
    assert PP.apply_policy_to_plan(one, 10, "HARD", CAPS) is not None
    assert PP.tyre_age_on_lap(plan, 8) == 8 and PP.tyre_age_on_lap(plan, 13) is None and PP.tyre_age_on_lap(plan, 14) == 2


def test_policy_validation():
    with pytest.raises(ValueError):
        PP.PitPolicy(1, 10, 5, "HARD")
    with pytest.raises(ValueError):
        PP.PitPolicy(5, 10, 5, "WET")
    text = PP.PitPolicy(14, 30, 10, "HARD").describe([Stint("MEDIUM", 24), Stint("HARD", 32)])
    assert "between laps 14 and 30" in text and "at least 10 laps old" in text and "Hards" in text and "M24-H32" in text


def run_with_masks(ctx, policy, n, seed):
    hook = PP.PolicyHook(policy, {ctx.hero: ctx.hero_plan}, ctx.tire_model, ctx.total_laps, ctx.caps)
    cap = {}

    def spy(sc, vsc, base):
        cap["sc"], cap["vsc"] = sc.copy(), vsc.copy()
        return hook(sc, vsc, base)

    simulate_grid(ctx.cars, ctx.base_laptime_s, ctx.tire_model, ctx.total_laps, n_sims=n, seed=seed,
                  pit_loss_s=ctx.pit_loss_s, safety_cars=ctx.safety_cars, plan_hook=spy)
    return hook, cap["sc"], cap["vsc"]


@pytest.mark.parametrize("trigger", ["sc", "sc_or_vsc"])
def test_policy_fires_only_on_sc_vsc_start_inside_window(toy_tire_model, trigger):
    ctx = make_ctx(toy_tire_model)
    pol = PP.PitPolicy(8, 20, 3, "HARD", trigger)
    hook, sc, vsc = run_with_masks(ctx, pol, 1500, 4)
    lap = hook.fire_lap[ctx.hero]
    fired = np.nonzero(lap)[0]
    assert len(fired) > 20 and len(fired) < 1500  # it fires sometimes, not always
    assert ((lap[fired] >= pol.window_lo) & (lap[fired] <= pol.window_hi)).all()
    for n_ in fired:
        i = lap[n_] - 1  # 0-indexed lap
        started_sc = sc[n_, i] and not sc[n_, i - 1]
        started_vsc = vsc[n_, i] and not vsc[n_, i - 1]
        assert started_sc or (trigger == "sc_or_vsc" and started_vsc)
        if trigger == "sc":
            assert started_sc and hook.fire_is_sc[ctx.hero][n_]
    # no event in the window -> no fire (checked from the masks, independent of the hook)
    none = np.nonzero(lap == 0)[0]
    starts = (sc[:, 1:] & ~sc[:, :-1]) if trigger == "sc" else ((sc[:, 1:] & ~sc[:, :-1]) | (vsc[:, 1:] & ~vsc[:, :-1]))
    in_win = starts[:, pol.window_lo - 2 : pol.window_hi - 1].any(axis=1)  # starts[:, i] is lap i+2
    # runs with a window event can still be silent (age < min_age or the reaction not allowed), but
    # runs with no window event must be silent
    assert not in_win[fired[:0]].any()
    assert (lap[~in_win] == 0).all()
    assert none.size > 0


def test_no_sc_means_never_fires_and_gain_is_zero(toy_tire_model):
    ctx = make_ctx(toy_tire_model, safety_cars=False)
    ev = PP.evaluate_policy(ctx, PP.PitPolicy(2, NL - 1, 3, "HARD"), 300, 1)
    assert ev.fire_rate == 0.0 and ev.gain_pts == 0.0 and ev.gain_pos == 0.0 and ev.gain_pts_ci == 0.0
    ctx_sc = make_ctx(toy_tire_model, safety_cars=True)
    ev2 = PP.evaluate_policy(ctx_sc, PP.PitPolicy(2, NL - 1, 3, "HARD"), 600, 1)
    assert ev2.fire_rate > 0.05  # sanity: with SCs it does fire


def test_deterministic_given_seed(toy_tire_model):
    ctx = make_ctx(toy_tire_model)
    pol = PP.PitPolicy(6, 24, 3, "HARD", "sc_or_vsc")
    a = PP.evaluate_policy(ctx, pol, 400, 11)
    b = PP.evaluate_policy(ctx, pol, 400, 11)
    c = PP.evaluate_policy(ctx, pol, 400, 12)
    assert a == b
    assert a != c
    s1 = PP.search_policy(ctx, n_screen=200, n_eval=300, seed=3, field_sensitivity=False)
    s2 = PP.search_policy(ctx, n_screen=200, n_eval=300, seed=3, field_sensitivity=False)
    assert s1.best == s2.best and s1.evaluation == s2.evaluation and s1.rule == s2.rule


def test_slot_tables_reproduce_the_policy_exactly(toy_tire_model):
    """The search's shortcut: a window policy's per-run outcome equals the outcome table of the first in-window event."""
    ctx = make_ctx(toy_tire_model)
    n, seed = 500, 5
    tables = PP.build_event_tables(ctx, n, seed, k_grid=(3,), compounds=("HARD",))
    slots = tables[(3, "HARD")]
    assert slots
    for lo, hi, trig in [(5, 25, "sc_or_vsc"), (10, 18, "sc"), (2, NL - 1, "sc_or_vsc")]:
        g_table, fired_table = PP.window_gain(slots, lo, hi, trig)
        pol = PP.PitPolicy(lo, hi, 3, "HARD", trig)
        base, _ = ctx.run(None, n, seed)
        res, hook = ctx.run(pol, n, seed)
        g_direct = PP.points_of(res, ctx.hero) - PP.points_of(base, ctx.hero)
        assert np.allclose(g_table, g_direct)
        assert np.array_equal(fired_table, hook.fire_lap[ctx.hero] > 0)


def test_empty_plan_hook_is_identical_to_no_hook(toy_tire_model):
    ctx = make_ctx(toy_tire_model)
    kw = dict(n_sims=200, seed=2, pit_loss_s=22.0, safety_cars=load_sc())
    a = simulate_grid(ctx.cars, BASE, toy_tire_model, NL, **kw)
    b = simulate_grid(ctx.cars, BASE, toy_tire_model, NL, plan_hook=lambda sc, vsc, t: {}, **kw)
    assert np.array_equal(a.positions, b.positions) and np.array_equal(a.finish_time_s, b.finish_time_s, equal_nan=True)
    # a hook that reproduces the car's own plan changes nothing either
    t, ag, p = PP._strategy_tables([ctx.cars[1].strategy], toy_tire_model, NL)
    nn = 200
    c = simulate_grid(ctx.cars, BASE, toy_tire_model, NL, plan_hook=lambda sc, vsc, tb: {
        1: (np.broadcast_to(t[0], (nn, NL)), np.broadcast_to(ag[0], (nn, NL)), np.broadcast_to(p[0], (nn, NL)))}, **kw)
    assert np.array_equal(a.positions, c.positions)


def test_unfired_runs_are_untouched(toy_tire_model):
    """Common random numbers: in runs where the rule does not fire, the hero (and everyone) finishes exactly as before."""
    ctx = make_ctx(toy_tire_model)
    pol = PP.PitPolicy(6, 24, 3, "HARD")
    base, _ = ctx.run(None, 500, 9)
    res, hook = ctx.run(pol, 500, 9)
    quiet = hook.fire_lap[ctx.hero] == 0
    assert quiet.sum() > 0
    assert np.array_equal(base.positions[quiet], res.positions[quiet])


# ------------------------------------------------------------------ API and tools

client = TestClient(app)


def test_api_smoke():
    r = client.post("/strategy/policy", json={"track_id": "austin", "n_sims": 200, "n_screen": 100,
                                              "field_sensitivity": False, "seed": 1})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["track_id"] == "austin" and d["driver"] and d["total_laps"] > 0
    assert sum(s["laps"] for s in d["base_plan"]) == d["total_laps"]
    assert isinstance(d["rule"], str) and d["rule"]
    if d["policy"]:
        p = d["policy"]
        assert 2 <= p["window_lo"] <= p["window_hi"] and p["compound"] in ("SOFT", "MEDIUM", "HARD")
        assert d["gain"]["points_ci95"] >= 0 and 0 <= d["gain"]["fire_rate"] <= 1
    r2 = client.post("/strategy/policy", json={"track_id": "austin", "n_sims": 200, "n_screen": 100,
                                               "field_sensitivity": False, "seed": 1})
    assert r2.json() == d  # deterministic


def test_api_errors():
    assert client.post("/strategy/policy", json={"track_id": "nowhere"}).status_code == 404
    r = client.post("/strategy/policy", json={"track_id": "austin", "driver": "ZZZ", "n_sims": 200, "n_screen": 100})
    assert r.status_code == 422
    r = client.post("/strategy/policy", json={"track_id": "austin", "n_sims": 200, "n_screen": 100,
                                              "base_plan": [{"compound": "HARD", "laps": 3}]})
    assert r.status_code == 422


def test_api_no_safety_cars_gives_no_rule():
    r = client.post("/strategy/policy", json={"track_id": "austin", "n_sims": 200, "n_screen": 100,
                                              "safety_cars": False, "field_sensitivity": False})
    assert r.status_code == 200
    d = r.json()
    assert d["policy"] is None and d["gain"] is None and d["significant"] is False


def test_tool_registered_in_mcp_and_chat():
    from app import mcp_server
    from app.services import chat_service

    assert "optimize_pit_policy" in [f.__name__ for f in mcp_server.TOOLS]
    assert "optimize_pit_policy" in [t["function"]["name"] for t in chat_service.TOOL_SCHEMAS]
    out = mcp_server.optimize_pit_policy("austin", n_sims=200, n_screen=100, seed=1)
    assert "error" not in out and out["caveats"] and out["rule"]
    assert "error" in mcp_server.optimize_pit_policy("austin", driver="ZZZ", n_sims=200, n_screen=100)
