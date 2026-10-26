"""T-045: predicted order by expected points, grid-slot DNF factor, API fields."""
import numpy as np
import pandas as pd
import pytest

from app.core import predictor as P
from app.services import prediction_service as S
from test_predictor import _synthetic_history


def test_rank_by_points_puts_dominant_favourite_first_despite_dnf_tail():
    # Dutch GP 2026 shape: favourite mean position 4.07 (DNF tail) vs a safe driver at 4.05.
    pts = np.array([20.3, 14.4, 0.0, 0.0])
    pos = np.array([4.07, 4.05, 15.0, 14.0])
    assert P.rank_by_points(pts, pos).tolist() == [1, 2, 4, 3]  # back markers tie on points -> pos tie-break


def test_simulate_returns_points_and_conditional_position():
    th = np.r_[3.0, np.zeros(9)]
    model = P.OutcomeModel(b_race=1.0, b_grid=0.0, a_quali=1.0)
    out = P.simulate(th, th, np.arange(1, 11, dtype=float), np.r_[0.4, np.zeros(9)], model, "grid", 20000, 1)
    assert out["exp_pos_cls"][0] < out["exp_pos"][0]  # the DNF tail lifts the mean
    assert out["exp_pts"][0] > 10.0 and out["exp_pts"].sum() == pytest.approx(P.FIA_POINTS[:10].sum(), rel=1e-6)


def test_grid_dnf_factor_only_in_grid_mode_and_front_row_lower():
    res, q = _synthetic_history()
    hist = P.prepare_history(res, q)
    ent = P.race_features(hist, 2026, 6, P.PredictorConfig(n_sims=200))
    cfg = P.PredictorConfig()
    off = P.dnf_probabilities(hist, 2026, 6, ent, cfg, use_grid=False)
    on = P.dnf_probabilities(hist, 2026, 6, ent, cfg, use_grid=True)
    assert on.sum() == pytest.approx(off.sum())  # field expectation unchanged
    # prequali predictions must not read the target race's grid
    t = P.predict_race(hist, 2026, 6, "prequali", P.PredictorConfig(n_sims=200))
    assert np.allclose(sorted(t["p_dnf"]), sorted(off))


def test_predict_race_has_pred_rank_consistent_with_points():
    res, q = _synthetic_history()
    hist = P.prepare_history(res, q)
    t = P.predict_race(hist, 2026, 6, "grid", P.PredictorConfig(n_sims=2000))
    o = t.sort_values("pred_rank")
    assert o["pred_rank"].tolist() == list(range(1, len(t) + 1))
    assert (np.diff(o["exp_pts"].to_numpy()) <= 1e-12).all()


def test_shape_exposes_points_and_falls_back_for_legacy_csv():
    codes = ["AAA", "BBB", "CCC", "DDD"]
    base = pd.DataFrame({"code": codes, "team": "T", "driver_id": codes, "grid": [1, 2, 3, 4],
                         "p_win": [.7, .2, .07, .03], "p_podium": [.8, .7, .6, .5], "p_top10": [1, 1, 1, 1],
                         "exp_pos": [3.2, 2.1, 2.8, 3.0], "p_dnf": [.2, .05, .05, .05]})
    legacy = S.shape(base, "grid", {"round": 1}, "now", True)
    assert legacy["headline"]["podium"] == ["BBB", "CCC", "DDD"]  # by exp_pos
    assert all(d["exp_points"] is None for d in legacy["drivers"])
    new = S.shape(base.assign(exp_pts=[20.0, 15.0, 6.0, 3.0]), "grid", {"round": 1}, "now", False)
    assert new["headline"]["podium"] == ["AAA", "BBB", "CCC"]
    assert {d["code"]: d["pred_rank"] for d in new["drivers"]} == {"AAA": 1, "BBB": 2, "CCC": 3, "DDD": 4}
