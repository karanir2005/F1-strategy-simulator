"""T-038: qualifying TTL for pending rounds, tolerant grid-file reader, stale_reason, start times."""
import importlib.util
import os
import time
from pathlib import Path

import pytest

from app.core import jolpica_client as J
from app.services import prediction_service as svc

ROOT = Path(__file__).resolve().parents[2]


def _script():
    spec = importlib.util.spec_from_file_location("_pr_t038", ROOT / "predict_race.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _payload(n):
    races = [{"season": "2026", "round": "18", "raceName": "X", "Circuit": {"circuitId": "a"}, "date": "2026-10-25",
              "QualifyingResults": [{"position": str(i)} for i in range(n)]}]
    return {"MRData": {"total": str(n), "RaceTable": {"Races": races}}}


def test_pending_quali_is_refetched_after_ttl(tmp_path, monkeypatch):
    monkeypatch.setattr(J, "RAW_DIR", tmp_path)
    calls = []
    monkeypatch.setattr(J, "_http_get", lambda p, o: (calls.append(1), _payload(10 if len(calls) == 1 else 22))[1])
    J.get_pages("2026/18/qualifying", ttl_s=J.QUALI_PENDING_TTL_S)
    J.get_pages("2026/18/qualifying", ttl_s=J.QUALI_PENDING_TTL_S)
    assert len(calls) == 1  # fresh within TTL
    f = next(tmp_path.glob("*qualifying*"))
    old = time.time() - 2 * J.QUALI_PENDING_TTL_S
    os.utime(f, (old, old))
    pages = J.get_pages("2026/18/qualifying", ttl_s=J.QUALI_PENDING_TTL_S)
    assert len(calls) == 2 and len(pages[0]["MRData"]["RaceTable"]["Races"][0]["QualifyingResults"]) == 22
    # offline never refetches even when expired
    os.utime(f, (old, old))
    J.get_pages("2026/18/qualifying", ttl_s=J.QUALI_PENDING_TTL_S, offline=True)
    assert len(calls) == 2


def test_quali_ttl_only_without_race_results(tmp_path, monkeypatch):
    monkeypatch.setattr(J, "RAW_DIR", tmp_path)
    seen = []
    monkeypatch.setattr(J, "get_pages", lambda path, ttl_s=None, offline=False: (seen.append(ttl_s), [_payload(0)])[1])
    J.qualifying(2026, 18)
    (tmp_path / "2026_18_results__o0.json").write_text("{}")
    J.qualifying(2026, 18)
    assert seen == [J.QUALI_PENDING_TTL_S, None]


def test_grid_file_tolerant(tmp_path):
    S = _script()
    f = tmp_path / "g.csv"
    f.write_text("Code; Grid\n LEC ;1\nham;PL\nver;pit\nnor; 4\n")
    g = S.read_grid_file(f)
    assert g["code"].tolist() == ["LEC", "HAM", "VER", "NOR"]
    assert g["grid"].tolist() == [1, 0, 0, 4]
    f.write_text("code,grid\nLEC,\n")
    with pytest.raises(SystemExit, match="LEC"):
        S.read_grid_file(f)


def test_stale_reason_and_short_stale_cache(monkeypatch):
    svc.clear_cache()
    n = {"live": 0}

    def boom():
        n["live"] += 1
        raise RuntimeError("jolpica down")

    monkeypatch.setattr(svc, "compute_live", boom)
    monkeypatch.setattr(svc, "latest_saved", lambda: {"mode": "pre_quali", "race": {}, "stale": True})
    a = svc.get_next(now=1000.0)
    assert "jolpica down" in a["stale_reason"]
    svc.get_next(now=1030.0)
    assert n["live"] == 1  # stale payload cached ~60 s
    svc.get_next(now=1100.0)
    assert n["live"] == 2
    svc.clear_cache()


def test_iso_utc():
    assert svc._iso_utc("2026-10-25", "19:00:00Z") == "2026-10-25T19:00:00Z"
    assert svc._iso_utc("2026-10-25", None) is None
