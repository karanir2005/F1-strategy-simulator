"""T-023: /predict endpoints. The network and the 10k-sim predictor are always mocked."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import prediction_service as svc

client = TestClient(app)

RACE = {"round": 18, "name": "United States GP", "circuit": "COTA", "date": "2026-10-25", "quali_date": "2026-10-24"}


def _table(mode="prequali"):
    codes = ["ANT", "HAM", "RUS", "LEC"]
    return pd.DataFrame(
        {
            "driver_id": ["antonelli", "hamilton", "russell", "leclerc"],
            "code": codes,
            "team": ["Mercedes", "Ferrari", "Mercedes", "Ferrari"],
            "grid": [float("nan")] * 4 if mode == "prequali" else [2.0, 1.0, 3.0, 4.0],
            "p_win": [0.3, 0.4, 0.2, 0.1],
            "p_podium": [0.7, 0.8, 0.9, 0.6],
            "p_top10": [1, 1, 1, 1],
            "exp_pos": [2.0, 1.5, 2.5, 4.0],
            "p_dnf": [0.05] * 4,
        }
    )


@pytest.fixture(autouse=True)
def isolate(monkeypatch, tmp_path):
    svc.clear_cache()
    monkeypatch.setattr(svc, "PRED_DIR", tmp_path)
    monkeypatch.setattr(svc, "BY_RACE_CSV", tmp_path / "by_race.csv")
    monkeypatch.setattr(svc, "EVAL_CSV", tmp_path / "missing.csv")
    monkeypatch.setattr(svc, "driver_names", lambda season=2026: {"ANT": "Kimi Antonelli"})
    yield
    svc.clear_cache()


def _live(monkeypatch, counter):
    def fake():
        counter.append(1)
        return (18, "pre_quali", False), svc.shape(_table(), "prequali", RACE, "2026-10-19T00:00:00+00:00", False) | {"mode": "pre_quali"}

    monkeypatch.setattr(svc, "compute_live", fake)


def test_next_live_sorted_and_cached(monkeypatch):
    calls = []
    _live(monkeypatch, calls)
    r = client.get("/predict/next")
    assert r.status_code == 200
    d = r.json()
    assert d["stale"] is False and d["mode"] == "pre_quali"
    assert [x["code"] for x in d["drivers"]] == ["HAM", "ANT", "RUS", "LEC"]
    assert d["drivers"][1]["name"] == "Kimi Antonelli" and d["drivers"][0]["name"] == "Hamilton"
    assert d["drivers"][0]["grid"] is None
    assert d["headline"] == {"winner": "HAM", "podium": ["HAM", "ANT", "RUS"]}  # T-042: by lowest exp_pos
    client.get("/predict/next")
    assert len(calls) == 1  # second call served from cache


def test_cache_expires(monkeypatch):
    # the TTL shortens on/after quali day (see test_ttl_shortens_on_quali_day); pin it so this test doesn't depend on today's date
    monkeypatch.setattr(svc, "_ttl", lambda payload: svc.CACHE_TTL_S)
    calls = []
    _live(monkeypatch, calls)
    svc.get_next(now=1000.0)
    svc.get_next(now=1000.0 + svc.CACHE_TTL_S - 1)
    assert len(calls) == 1
    svc.get_next(now=1000.0 + svc.CACHE_TTL_S + 1)
    assert len(calls) == 2


def test_ttl_shortens_on_quali_day():
    past = {"mode": "pre_quali", "race": {**RACE, "quali_date": "2000-01-01"}}
    future = {"mode": "pre_quali", "race": {**RACE, "quali_date": "2999-01-01"}}
    grid = {"mode": "grid", "race": {**RACE, "quali_date": "2000-01-01"}}
    assert svc._ttl(past) == svc.QUALI_DAY_TTL_S
    assert svc._ttl(future) == svc.CACHE_TTL_S
    assert svc._ttl(grid) == svc.CACHE_TTL_S


def test_grid_mode_has_grid(monkeypatch):
    p = svc.shape(_table("grid"), "grid", RACE, "x", False)
    assert [d["grid"] for d in p["drivers"]] == [1, 2, 3, 4]


def test_fallback_to_newest_csv_stale(monkeypatch, tmp_path):
    def boom():
        raise RuntimeError("network down")

    monkeypatch.setattr(svc, "compute_live", boom)
    for name in ("2026_R18_prequali_a.csv", "2026_R18_prequali_b.csv"):
        t = _table()
        t["round"], t["mode"] = 18, "prequali"
        t.to_csv(tmp_path / name, index=False)
    r = client.get("/predict/next")
    assert r.status_code == 200 and r.json()["stale"] is True
    assert r.json()["race"]["round"] == 18 and r.json()["mode"] == "pre_quali"


def test_fallback_on_systemexit(monkeypatch, tmp_path):
    def exits():
        raise SystemExit("grid mode: no qualifying")

    monkeypatch.setattr(svc, "compute_live", exits)
    t = _table()
    t["round"], t["mode"] = 18, "prequali"
    t.to_csv(tmp_path / "2026_R18_prequali_a.csv", index=False)
    assert client.get("/predict/next").json()["stale"] is True


def test_503_when_nothing(monkeypatch):
    def boom():
        raise RuntimeError("down")

    monkeypatch.setattr(svc, "compute_live", boom)
    r = client.get("/predict/next")
    assert r.status_code == 503 and "no saved" in r.json()["detail"]


def test_history(tmp_path, monkeypatch):
    monkeypatch.setattr(svc, "_pole_sitters", lambda: {1: "RUS"})
    pd.DataFrame(
        [
            {"round": 1, "race": "Australian Grand Prix", "pred_winner": "RUS", "pred_winner_p": 0.5, "actual_winner": "RUS",
             "actual_winner_grid": 1, "p_actual_winner": 0.5, "pred_podium": "RUS ANT HAD", "actual_podium": "RUS ANT LEC", "podium_hits": 2}
        ]
    ).to_csv(tmp_path / "by_race.csv", index=False)
    r = client.get("/predict/history")
    assert r.status_code == 200
    row = r.json()["races"][0]
    assert row["pole_sitter"] == "RUS" and row["winner_correct"] is True
    assert row["predicted_podium"] == ["RUS", "ANT", "HAD"] and row["podium_hits"] == 2


def test_history_503(monkeypatch):
    assert client.get("/predict/history").status_code == 503


def test_cors_header():
    r = client.options("/predict/next", headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"
