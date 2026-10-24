"""T-029: pre-quali comparison fields on the grid-mode prediction payload."""

import pandas as pd

from app.services import prediction_service as svc


def _table(codes):
    return pd.DataFrame({
        "code": codes, "team": ["T"] * len(codes), "driver_id": [c.lower() for c in codes],
        "grid": list(range(1, len(codes) + 1)), "p_win": [0.5, 0.3, 0.2][: len(codes)],
        "p_podium": [0.9, 0.8, 0.7][: len(codes)], "p_top10": [1.0] * len(codes),
        "exp_pos": [1.0, 2.0, 3.0][: len(codes)], "p_dnf": [0.05] * len(codes),
    })


def _race(rnd=18):
    return {"round": rnd, "name": "x", "circuit": "", "date": "", "quali_date": None}


def _csv(path, rows):
    pd.DataFrame(rows, columns=["code", "p_win", "p_podium"]).to_csv(path, index=False)


def test_prequali_fields_use_newest_file_and_null_for_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(svc, "PRED_DIR", tmp_path)
    _csv(tmp_path / "2026_R18_prequali_20261019_1533.csv", [("AAA", 0.9, 0.9), ("BBB", 0.1, 0.1)])
    _csv(tmp_path / "2026_R18_prequali_20261019_1714.csv", [("AAA", 0.4, 0.8), ("BBB", 0.35, 0.6)])
    _csv(tmp_path / "2026_R17_prequali_20261020_0900.csv", [("AAA", 0.0, 0.0)])  # other round, ignored
    p = svc.shape(_table(["AAA", "BBB", "CCC"]), "grid", _race(), "t", False, "override_file")
    d = {x["code"]: x for x in p["drivers"]}
    assert d["AAA"]["p_win_prequali"] == 0.4 and d["AAA"]["p_podium_prequali"] == 0.8
    assert d["BBB"]["p_win_prequali"] == 0.35
    assert d["CCC"]["p_win_prequali"] is None and d["CCC"]["p_podium_prequali"] is None


def test_prequali_fields_null_without_file_and_absent_in_prequali_mode(monkeypatch, tmp_path):
    monkeypatch.setattr(svc, "PRED_DIR", tmp_path)
    p = svc.shape(_table(["AAA", "BBB"]), "grid", _race(), "t", False)
    assert all(d["p_win_prequali"] is None and d["p_podium_prequali"] is None for d in p["drivers"])
    _csv(tmp_path / "2026_R18_prequali_20261019_1714.csv", [("AAA", 0.4, 0.8)])
    q = svc.shape(_table(["AAA", "BBB"]), "prequali", _race(), "t", False)
    assert all("p_win_prequali" not in d for d in q["drivers"])


# ---- track-level DNF component (off by default)

import numpy as np
from app.core import predictor as P


def _hist(rows):
    res = pd.DataFrame(rows, columns=["season", "round", "circuit_id", "classified", "team_id", "driver_id"])
    return {"results": res}


def _rows():
    out = []
    for s in (2024, 2025):  # same circuit "a" in round 1, "b" in round 2; 10 cars each
        for rnd, cid, nf in ((1, "a", 4 if s == 2024 else 3), (2, "b", 1)):
            for i in range(10):
                out.append((s, rnd, cid, i >= nf, "t", f"d{i}"))
    out.append((2026, 1, "a", True, "t", "d0"))  # target race itself: must be ignored
    return out


def test_track_dnf_off_by_default_and_unity_for_unknown_track():
    h = _hist(_rows())
    assert P.PredictorConfig().dnf_track_m == 0.0
    assert P.track_dnf_ratio(h, 2026, 1, P.PredictorConfig()) == 1.0
    assert P.track_dnf_ratio(h, 2026, 5, P.PredictorConfig(dnf_track_m=4.0)) == 1.0  # circuit_of -> None


def test_track_dnf_ratio_shrinks_and_ignores_target_race():
    h = _hist(_rows())
    cfg = P.PredictorConfig(dnf_track_m=4.0)
    r = P.track_dnf_ratio(h, 2026, 1, cfg)
    assert r > 1.0  # circuit "a" lost 7 of 20 starts vs a ~20-25% season rate
    # more shrinkage -> closer to 1; target-season rows can't change the answer
    assert 1.0 < P.track_dnf_ratio(h, 2026, 1, P.PredictorConfig(dnf_track_m=40.0)) < r
    h2 = _hist(_rows()[:-1] + [(2026, 1, "a", False, "t", "d0")])
    assert P.track_dnf_ratio(h2, 2026, 1, cfg) == r


def test_dnf_probabilities_scaled_only_when_on():
    h = _hist(_rows())
    h["results"]["team_id"] = "t"
    ent = pd.DataFrame({"driver_id": ["d0", "d1"], "team_id": ["t", "t"]})
    off = P.dnf_probabilities(h, 2026, 1, ent, P.PredictorConfig())
    on = P.dnf_probabilities(h, 2026, 1, ent, P.PredictorConfig(dnf_track_m=4.0))
    assert np.all(on > off) and np.all(on <= 0.9)
