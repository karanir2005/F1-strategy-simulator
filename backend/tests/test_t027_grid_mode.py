"""T-027: grid-mode switch, grid override file, and CLI/API consistency.

No network and no writes to data/jolpica: Jolpica is replaced by the cached 2026 tables (offline)
plus a fabricated round-18 qualifying copied from round 17, the override directory and the
prediction output directory are tmp_path, and n_sims is small.
"""

import argparse
from datetime import datetime, timezone

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import prediction_service as svc

client = TestClient(app)
S = svc._predict_script()

R17_CODES = None


def _grid_rows(pit_lane_code="RUS"):
    """22-driver post-penalty grid; `pit_lane_code` starts from the pit lane (0)."""
    global R17_CODES
    if R17_CODES is None:
        r = pd.read_csv(svc.REPO_ROOT / "data" / "jolpica" / "derived" / "results.csv")
        R17_CODES = r[(r["season"] == 2026) & (r["round"] == 17)].sort_values("grid")["code"].tolist()
    codes = [c for c in R17_CODES if c != pit_lane_code]
    rows = [(c, i + 1) for i, c in enumerate(codes)] + [(pit_lane_code, 0)]
    return pd.DataFrame(rows, columns=["code", "grid"])


@pytest.fixture
def env(monkeypatch, tmp_path):
    svc.clear_cache()
    state = {"quali": False}
    orig = S.J.load_all

    def fake_load_all(*a, **k):
        k.update(offline=True, write_csv=False)
        res, qual, spr, sched = orig(*a, **k)
        qual = qual[~((qual["season"] == 2026) & (qual["round"] == 18))]
        if state["quali"]:
            fab = qual[(qual["season"] == 2026) & (qual["round"] == 17)].copy()
            fab["round"] = 18
            qual = pd.concat([qual, fab], ignore_index=True)
        return res, qual, spr, sched

    monkeypatch.setattr(S.J, "load_all", fake_load_all)
    monkeypatch.setattr(S, "next_round", lambda offline: 18)
    monkeypatch.setattr(S, "CONFIG", S.P.PredictorConfig(**{**S.CONFIG.__dict__, "n_sims": 400}))
    monkeypatch.setattr(svc, "OVERRIDE_DIR", tmp_path / "ov")
    (tmp_path / "ov").mkdir()
    monkeypatch.setattr(svc, "PRED_DIR", tmp_path / "preds")
    (tmp_path / "preds").mkdir()
    monkeypatch.setattr(svc, "EVAL_CSV", tmp_path / "missing.csv")
    yield state
    svc.clear_cache()


def _override(tmp_path, df):
    df.to_csv(tmp_path / "ov" / "grid_override_2026_R18.csv", index=False)


def test_prequali_when_no_qualifying(env):
    d = client.get("/predict/next").json()
    assert d["mode"] == "pre_quali" and d["grid_source"] is None
    assert all(x["grid"] is None for x in d["drivers"])


def test_switches_to_grid_when_qualifying_published(env):
    assert client.get("/predict/next").json()["mode"] == "pre_quali"
    env["quali"] = True
    svc.clear_cache()  # stands in for the TTL expiring
    d = client.get("/predict/next").json()
    assert d["mode"] == "grid" and d["grid_source"] == "qualifying_classification"
    assert sorted(x["grid"] for x in d["drivers"])[0] == 1
    assert not d["stale"]
    assert any("penalties are not applied" in n for n in d["evidence"]["notes"])


def test_override_file_wins_over_classification(env, tmp_path):
    env["quali"] = True
    g = _grid_rows("RUS")
    _override(tmp_path, g)
    d = client.get("/predict/next").json()
    assert d["mode"] == "grid" and d["grid_source"] == "override_file"
    got = {x["code"]: x["grid"] for x in d["drivers"]}
    assert got == dict(zip(g["code"], g["grid"]))
    assert got["RUS"] == 0  # pit-lane starter
    assert any("override" in n for n in d["evidence"]["notes"])


def test_override_alone_switches_to_grid_mode(env, tmp_path):
    _override(tmp_path, _grid_rows("HAM"))
    d = client.get("/predict/next").json()
    assert d["mode"] == "grid" and d["grid_source"] == "override_file"


def test_bad_override_falls_back_to_stale_csv(env, tmp_path):
    g = _grid_rows()
    g.loc[1, "grid"] = g.loc[0, "grid"]  # duplicate grid slot -> SystemExit in the loader
    _override(tmp_path, g)
    t = pd.DataFrame({"driver_id": ["a"], "code": ["ANT"], "team": ["M"], "p_win": [1.0], "p_podium": [1.0], "p_top10": [1.0],
                      "exp_pos": [1.0], "p_dnf": [0.0], "round": [18], "mode": ["prequali"]})
    t.to_csv(tmp_path / "preds" / "2026_R18_prequali_x.csv", index=False)
    d = client.get("/predict/next").json()
    assert d["stale"] is True and d["grid_source"] is None


def test_cli_and_api_agree_and_csv_roundtrips_to_stale(env, tmp_path, capsys, monkeypatch):
    g = _grid_rows("RUS")
    gf = tmp_path / "grid.csv"
    g.to_csv(gf, index=False)
    args = argparse.Namespace(round=18, offline=True, fp=False, mode="grid", grid_file=str(gf), sims=400,
                              out_dir=str(tmp_path / "preds"))
    S.predict(args)
    out = capsys.readouterr().out
    assert "mode: grid" in out and "grid_source: override_file" in out
    assert "sums: win 1.000, podium 3.000, top10 10.000" in out
    files = list((tmp_path / "preds").glob("2026_R18_grid_*.csv"))
    assert len(files) == 1  # naming: 2026_R18_<mode>_<stamp>.csv, mode is the CLI spelling
    saved = pd.read_csv(files[0])
    assert (saved["grid_source"] == "override_file").all() and int(saved["round"].iloc[0]) == 18
    hl = S.headline(saved)
    assert f"headline: winner {hl['winner']}, podium {' '.join(hl['podium'])}" in out

    # API from the same inputs: same mode and (seeded, same n_sims) same headline.
    _override(tmp_path, g)
    api = client.get("/predict/next").json()
    assert api["mode"] == "grid" and api["grid_source"] == "override_file"
    assert api["headline"] == hl

    # Stale fallback reads the CLI-written CSV and keeps the penalty note.
    def boom():
        raise RuntimeError("down")

    monkeypatch.setattr(svc, "compute_live", boom)
    svc.clear_cache()
    st = client.get("/predict/next").json()
    assert st["stale"] is True and st["mode"] == "grid" and st["grid_source"] == "override_file"
    assert st["headline"] == hl


def test_legacy_csv_without_grid_source_column(env, tmp_path, monkeypatch):
    monkeypatch.setattr(svc, "compute_live", lambda: (_ for _ in ()).throw(RuntimeError("down")))
    t = pd.DataFrame({"driver_id": ["a"], "code": ["ANT"], "team": ["M"], "grid": [1.0], "p_win": [1.0], "p_podium": [1.0],
                      "p_top10": [1.0], "exp_pos": [1.0], "p_dnf": [0.0], "round": [18], "mode": ["grid"]})
    t.to_csv(tmp_path / "preds" / "2026_R18_grid_old.csv", index=False)
    d = client.get("/predict/next").json()
    assert d["stale"] and d["mode"] == "grid" and d["grid_source"] is None


def test_quali_day_ttl_is_short_only_for_prequali():
    today = datetime.now(timezone.utc).date().isoformat()
    pre = {"mode": "pre_quali", "race": {"quali_date": today}}
    assert svc._ttl(pre) == svc.QUALI_DAY_TTL_S
    assert svc._ttl({"mode": "pre_quali", "race": {"quali_date": "2999-01-01"}}) == svc.CACHE_TTL_S
    assert svc._ttl({"mode": "grid", "race": {"quali_date": today}}) == svc.CACHE_TTL_S


def test_override_path_name():
    assert S.override_grid_path(18).name == "grid_override_2026_R18.csv"
