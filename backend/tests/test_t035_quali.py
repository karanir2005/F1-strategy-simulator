"""T-035: qualifying distribution from simulate(), walk-forward helpers, and the API `qualifying` block."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.core import predictor as P
from app.main import app
from app.services import prediction_service as svc

client = TestClient(app)
RACE = {"round": 18, "name": "United States GP", "circuit": "COTA", "date": "2026-10-25", "quali_date": "2026-10-24"}


def test_q1_cutoff_rule():
    assert P.q1_cutoff(22) == 17  # 6 out
    assert P.q1_cutoff(20) == 16  # 5 out
    assert P.q1_cutoff(10) == 11  # nobody out


def test_simulate_quali_events_consistent_and_race_numbers_unchanged():
    n = 22
    theta = np.linspace(2, -2, n)
    m = P.OutcomeModel(b_race=1.0, b_grid=0.5, a_quali=1.0)
    s = P.simulate(theta, theta, None, np.full(n, 0.05), m, "prequali", 4000, 1)
    assert s["p_pole"].sum() == pytest.approx(1)
    assert s["p_front_row"].sum() == pytest.approx(2)
    assert s["p_q3"].sum() == pytest.approx(10)
    assert s["p_q1_exit"].sum() == pytest.approx(6)
    assert s["p_pole"][0] > s["p_pole"][-1] and s["p_q1_exit"][-1] > s["p_q1_exit"][0]
    assert s["exp_grid"][0] < s["exp_grid"][-1]
    # grid mode has no qualifying keys; the race outputs use the same RNG stream as before
    g = P.simulate(theta, theta, np.arange(1, n + 1.0), np.full(n, 0.05), m, "grid", 100, 1)
    assert "p_pole" not in g


def _table():
    return pd.DataFrame(
        {
            "driver_id": ["a", "b", "c"], "code": ["ANT", "HAM", "RUS"], "team": ["M", "F", "M"],
            "grid": [2.0, 1.0, 3.0], "p_win": [0.3, 0.5, 0.2], "p_podium": [0.7, 0.8, 0.9], "p_top10": [1, 1, 1],
            "exp_pos": [2.0, 1.5, 2.5], "p_dnf": [0.05] * 3,
        }
    )


QCOLS = {"p_pole": [0.2, 0.5, 0.3], "p_front_row": [0.5, 0.8, 0.7], "p_q3": [0.9, 0.95, 0.9],
         "p_q1_exit": [0.02, 0.01, 0.02], "exp_grid": [2.2, 1.6, 2.4]}


@pytest.fixture(autouse=True)
def isolate(monkeypatch, tmp_path):
    svc.clear_cache()
    monkeypatch.setattr(svc, "PRED_DIR", tmp_path)
    monkeypatch.setattr(svc, "driver_names", lambda season=2026: {})
    monkeypatch.setattr(svc, "QUALI_EVAL_CSV", tmp_path / "q_eval.csv")
    yield
    svc.clear_cache()


def _eval_csv(tmp_path):
    pd.DataFrame(
        [
            {"season": 2026, "method": "model", "pole_logloss": 0.13, "q3_logloss": 0.3, "spearman": 0.85, "n_races": 16},
            {"season": 2026, "method": "champ", "pole_logloss": 0.14, "q3_logloss": 0.36, "spearman": 0.82, "n_races": 16},
            {"season": 2026, "method": "prev_quali", "pole_logloss": 0.15, "q3_logloss": 0.4, "spearman": 0.79, "n_races": 16},
            {"season": 2026, "method": "uniform", "pole_logloss": 0.19, "q3_logloss": 0.69, "spearman": np.nan, "n_races": 16},
        ]
    ).to_csv(tmp_path / "q_eval.csv", index=False)


def test_prequali_block_from_table(tmp_path):
    _eval_csv(tmp_path)
    t = _table().assign(**QCOLS)
    p = svc.shape(t, "prequali", RACE, "x", False)
    q = p["qualifying"]
    assert q["available"] and [d["code"] for d in q["drivers"]] == ["HAM", "RUS", "ANT"]
    assert all(d["actual_grid"] is None for d in q["drivers"])
    assert q["evidence"]["baseline_name"] == "championship order" and q["evidence"]["races"] == 16
    assert q["evidence"]["pole_logloss"] < q["evidence"]["baseline_pole_logloss"]


def test_prequali_without_columns_is_null():
    assert svc.shape(_table(), "prequali", RACE, "x", False)["qualifying"] is None


def test_grid_mode_uses_saved_file_and_fills_actual(tmp_path):
    t = _table().assign(**QCOLS)[["code"] + list(QCOLS)]
    t.to_csv(tmp_path / "2026_R18_prequali_20261023_1000.csv", index=False)
    old = t.drop(columns=["p_pole"])  # newer file without the columns is skipped
    old.to_csv(tmp_path / "2026_R18_prequali_20261023_1100.csv", index=False)
    q = svc.shape(_table(), "grid", RACE, "x", False, "qualifying_classification")["qualifying"]
    by = {d["code"]: d for d in q["drivers"]}
    assert by["HAM"]["actual_grid"] == 1 and by["ANT"]["actual_grid"] == 2 and by["HAM"]["p_pole"] == 0.5
    assert q["evidence"] is None  # eval file missing


def test_grid_mode_no_saved_file_is_null():
    assert svc.shape(_table(), "grid", RACE, "x", False)["qualifying"] is None


def test_http_keeps_qualifying_and_t029_fields(monkeypatch, tmp_path):
    t = _table().assign(**QCOLS)[["code"] + list(QCOLS)]
    t.assign(p_win=0.3, p_podium=0.6).to_csv(tmp_path / "2026_R18_prequali_20261023_1000.csv", index=False)
    payload = svc.shape(_table(), "grid", RACE, "x", False) | {"mode": "grid"}
    monkeypatch.setattr(svc, "compute_live", lambda: ((18, "grid", True, None), payload))
    d = client.get("/predict/next").json()
    assert d["qualifying"]["drivers"][0]["actual_grid"] in (1, 2, 3)
    assert d["drivers"][0]["p_win_prequali"] == 0.3
    # pre-quali payloads keep the T-029 fields absent and old clients unaffected
    svc.clear_cache()
    pre = svc.shape(_table(), "prequali", RACE, "x", False) | {"mode": "pre_quali"}
    monkeypatch.setattr(svc, "compute_live", lambda: ((18, "pre_quali", False, None), pre))
    d = client.get("/predict/next").json()
    assert d["qualifying"] is None and "p_win_prequali" not in d["drivers"][0]


def test_walk_forward_baselines_synthetic():
    rows_q, rows_r = [], []
    ids = [f"d{i}" for i in range(12)]
    for s in (2022, 2023, 2024, 2025, 2026):
        for r in range(1, 5):
            for i, d in enumerate(ids):
                rows_q.append({"season": s, "round": r, "driver_id": d, "quali_position": i + 1})
                rows_r.append({"season": s, "round": r, "driver_id": d, "points": float(12 - i)})
    hist = {"qualifying": pd.DataFrame(rows_q), "results": pd.DataFrame(rows_r)}
    for m in P.QUALI_BASELINES:
        sc = P.quali_score(P.quali_walk_forward(hist, 2026, range(2, 5), m))
        assert sc["n_races"] == 3
    # perfectly stable order: prev_quali beats uniform on pole log-loss, Spearman = 1
    a = P.quali_score(P.quali_walk_forward(hist, 2026, range(2, 5), "prev_quali"))
    u = P.quali_score(P.quali_walk_forward(hist, 2026, range(2, 5), "uniform"))
    assert a["pole_logloss"] < u["pole_logloss"] and a["spearman"] == pytest.approx(1)
