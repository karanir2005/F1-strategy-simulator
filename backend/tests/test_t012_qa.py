"""
T-012 QA tests for the T-011 predictor. Work from the cached Jolpica data
(offline, skipped if absent); nothing is written to the repo.
"""

import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.core import jolpica_client as J
from app.core import predictor as P

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

pytestmark = pytest.mark.skipif(
    not (REPO / "data" / "jolpica" / "raw" / "2026_17_results__o0.json").exists(),
    reason="cached Jolpica data not present",
)


@pytest.fixture(scope="module")
def real_hist(tmp_path_factory):
    J.DERIVED_DIR = tmp_path_factory.mktemp("derived")
    res, q, spr, sched = J.load_all(current_upto_round=17, offline=True, write_csv=False)
    return P.prepare_history(res, q, sprint=spr), sched


def _truncate(hist, season, rnd):
    """Physically remove the target race and everything later."""
    out = {}
    for k, df in hist.items():
        out[k] = df[~((df["season"] == season) & (df["round"] >= rnd))].reset_index(drop=True)
    return out


@pytest.mark.parametrize("rnd", [5, 12])
@pytest.mark.parametrize("mode", ["grid", "prequali"])
def test_prediction_unchanged_when_target_and_future_physically_removed(real_hist, rnd, mode):
    """Stronger than the scramble test: delete the target/future rows entirely
    (keeping only entry list + grid, supplied explicitly) and require the same
    probabilities as the full-history call."""
    hist, _ = real_hist
    cfg = P.PredictorConfig(n_sims=3000)
    full = P.predict_race(hist, 2026, rnd, mode, cfg)
    # same entrant order as the full call (the MC draws depend on column order)
    ent = P.entrants_for(hist, 2026, rnd)
    grid = None
    if mode == "grid":
        g = hist["results"].query("season == 2026 and round == @rnd").set_index("driver_id")["grid"]
        grid = ent["driver_id"].map(g).to_numpy(dtype=float)
    trunc = P.predict_race(_truncate(hist, 2026, rnd), 2026, rnd, mode, cfg, entry_list=ent, grid=grid)
    a = full.set_index("driver_id")[["p_win", "p_podium", "p_top10", "p_dnf"]]
    b = trunc.set_index("driver_id")[["p_win", "p_podium", "p_top10", "p_dnf"]].loc[a.index]
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), atol=1e-12)


def test_sums_real_data_all_rounds(real_hist):
    hist, _ = real_hist
    cfg = P.PredictorConfig(n_sims=2000)
    for mode in ("grid", "prequali"):
        for rnd in (2, 9, 17):
            t = P.predict_race(hist, 2026, rnd, mode, cfg)
            assert t["p_win"].sum() == pytest.approx(1.0, abs=1e-9)
            assert t["p_podium"].sum() == pytest.approx(3.0, abs=1e-9)
            assert t["p_top10"].sum() == pytest.approx(10.0, abs=1e-9)
            assert t[["p_win", "p_podium", "p_top10"]].min().min() >= 0


def _args(**kw):
    d = dict(round=18, mode="auto", grid_file=None, offline=True, sims=2000, fp=False)
    d.update(kw)
    return Namespace(**d)


@pytest.fixture
def pr(tmp_path, monkeypatch):
    import predict_race as pr_

    monkeypatch.setattr(pr_, "OUT_DIR", tmp_path)
    monkeypatch.setattr(J, "DERIVED_DIR", tmp_path / "derived")
    return pr_


def _hist18(pr, extra_quali=None):
    hist, _, _ = pr.load(True, 18)
    if extra_quali is not None:
        hist["qualifying"] = pd.concat([hist["qualifying"], extra_quali], ignore_index=True)
    return hist


def _fake_quali18(hist, drop=()):
    q = hist["qualifying"]
    q = q[(q["season"] == 2026) & (q["round"] == 17)].copy()
    q["round"] = 18
    return q[~q["code"].isin(drop)]


def test_grid_file_path_end_to_end(pr, tmp_path, capsys):
    """--mode grid --grid-file with a pit-lane start (0), lowercase code."""
    hist = _hist18(pr)
    ent = P.entrants_for(hist, 2026, 17)
    codes = ent["code"].tolist()
    g = pd.DataFrame({"code": [c.lower() for c in codes], "grid": range(1, len(codes) + 1)})
    g.loc[g["code"] == "ver", "grid"] = 0  # pit-lane start
    p = tmp_path / "grid.csv"
    g.to_csv(p, index=False)
    pr.predict(_args(mode="grid", grid_file=str(p)))
    out = capsys.readouterr().out
    assert "grid from" in out and "sums: win 1.000, podium 3.000, top10 10.000" in out
    saved = list((tmp_path / "predictions").glob("2026_R18_grid_*.csv"))
    assert len(saved) == 1
    t = pd.read_csv(saved[0])
    assert len(t) == 22
    # pit-lane starter is treated as last on the grid -> low win probability
    assert t.set_index("code").loc["VER", "p_win"] < 0.05


def test_grid_file_unknown_code_stops_clearly(pr, tmp_path):
    hist = _hist18(pr)
    codes = P.entrants_for(hist, 2026, 17)["code"].tolist()
    g = pd.DataFrame({"code": codes[:-1] + ["XXX"], "grid": range(1, len(codes) + 1)})
    p = tmp_path / "grid.csv"
    g.to_csv(p, index=False)
    with pytest.raises(SystemExit, match="XXX"):
        pr.predict(_args(mode="grid", grid_file=str(p)))


def test_auto_switches_to_grid_when_round18_quali_exists(pr, monkeypatch, capsys):
    real_load = pr.load

    def fake_load(offline, rnd, with_fp=False):
        hist, spr, sched = real_load(offline, rnd, with_fp)
        hist["qualifying"] = pd.concat([hist["qualifying"], _fake_quali18(hist)], ignore_index=True)
        return hist, spr, sched

    monkeypatch.setattr(pr, "load", fake_load)
    pr.predict(_args(mode="auto"))
    out = capsys.readouterr().out
    assert "mode: grid" in out and "qualifying for this round AVAILABLE" in out
    assert "NO penalties applied" in out


def test_auto_without_quali_is_prequali(pr, capsys):
    pr.predict(_args(mode="auto"))
    assert "mode: prequali" in capsys.readouterr().out


def test_drivers_missing_from_jolpica_quali_still_in_field(pr, monkeypatch, capsys):
    """T-012 finding 1 (fixed). Jolpica's qualifying table omits drivers
    who set no time (2026 R1: VER, SAI, STR; R14: BEA, STR; all started the
    race). Auto/grid mode takes the entry list from that table, so such a
    driver silently disappears from the field. The code comment says they
    should 'start from the back'."""
    real_load = pr.load

    def fake_load(offline, rnd, with_fp=False):
        hist, spr, sched = real_load(offline, rnd, with_fp)
        hist["qualifying"] = pd.concat(
            [hist["qualifying"], _fake_quali18(hist, drop=("VER", "STR"))], ignore_index=True
        )
        return hist, spr, sched

    monkeypatch.setattr(pr, "load", fake_load)
    pr.predict(_args(mode="auto"))
    saved = list((pr.OUT_DIR / "predictions").glob("2026_R18_grid_*.csv"))
    t = pd.read_csv(saved[0])
    assert len(t) == 22 and {"VER", "STR"} <= set(t["code"])
    assert "WARNING" in capsys.readouterr().out
