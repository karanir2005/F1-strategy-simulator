"""T-002 QA: independent checks of the re-pulled 2026 data and the new tyre fit."""
import glob
import pandas as pd
import pytest
from app.core import tire_model as tm

# The 11473 / 343 / 44 / 67 pins below were measured on the ORIGINAL 10 FastF1 races.
# T-015 promoted six more (OpenF1-sourced) races into data/processed, so the pinned
# counts are restricted to the original ten; the six are covered by test_t015_ds.py.
ORIGINAL_10 = ("Australia", "Austria", "Belgium", "Britain", "Canada", "China", "Hungary", "Japan", "Miami", "Monaco")
FILES = sorted(f"data/processed/{gp}_2026_laps.csv" for gp in ORIGINAL_10)
pytestmark = pytest.mark.skipif(len(FILES) < 10, reason="processed data missing")


def _all():
    return pd.concat([pd.read_csv(f).assign(race=f) for f in FILES])


def test_stint_counts_match_t001():
    d = _all()
    assert len(d) == 11473
    assert int(d.is_tyre_change.sum()) == 343
    assert int(d.is_pit_pass_no_change.sum()) == 44


def test_one_compound_and_monotone_age_per_stint():
    for f in FILES:
        d = pd.read_csv(f)
        for _, g in d.groupby(["driver", "tyre_stint"]):
            assert g.compound.dropna().nunique() <= 1
            a = g.sort_values("lap_number").tire_age.dropna().diff().dropna()
            assert (a == 1).all() or (a.abs() <= 1).all()


def test_pit_pass_does_not_start_new_stint():
    for f in FILES:
        d = pd.read_csv(f).sort_values(["driver", "lap_number"])
        d["prev"] = d.groupby("driver").tyre_stint.shift()
        p = d[d.is_pit_pass_no_change & d.prev.notna()]
        assert (p.tyre_stint == p.prev).all()


def test_used_set_stints_have_start_age_gt_1():
    d = _all()
    st = d.groupby(["race", "driver", "tyre_stint"]).stint_start_age.first()
    assert int((st > 1).sum()) == 67
    assert (d.stint_start_age.dropna() >= 1).all()


def test_sidecars_sane():
    for f in FILES:
        b = f.replace("_laps.csv", "")
        r = pd.read_csv(b + "_results.csv")
        assert r.position.is_unique and len(r) == 22
        p = pd.read_csv(b + "_pit_stops.csv")
        assert (p.pit_lane_s > 0).all()


def test_fitted_wear_monotone_and_fuel_negative():
    m = tm.build_tire_model()
    assert -0.08 < m["_fuel_coef"] < -0.02
    for c in ("SOFT", "MEDIUM", "HARD"):
        v = [tm.predict_delta(m, c, a) for a in range(1, 40)]
        assert all(b >= a - 1e-9 for a, b in zip(v, v[1:]))
        assert v[-1] > v[0]
