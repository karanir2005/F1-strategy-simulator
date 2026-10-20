"""T-015/T-020: era filter, ordered tyre fit, promoted races (reads data/processed)."""
import glob

import pandas as pd
import pytest

from app.core import tire_model as tm

SIX = ("Barcelona", "Madrid", "Italy", "Azerbaijan", "Singapore", "Netherlands")


def test_six_races_promoted_and_16_total():
    files = glob.glob("data/processed/*_2026_laps.csv")
    assert len(files) == 16
    for gp in SIX:
        assert any(f"{gp}_2026_laps" in f for f in files)
    assert not any("Malaysia" in f for f in files)


def test_era_filter_drops_2025_by_default():
    l26 = tm.load_all_laps()
    assert set(tm.lap_eras(l26)) == {tm.ERA_2026}
    l25 = tm.load_2025_staging_laps()
    assert set(tm.lap_eras(l25)) == {tm.ERA_2025}
    mixed = pd.concat([l26, l25], ignore_index=True)
    assert set(tm.lap_eras(tm.filter_eras(mixed))) == {tm.ERA_2026}
    assert set(tm.lap_eras(tm.filter_eras(mixed, None))) == {tm.ERA_2025, tm.ERA_2026}


def test_default_model_is_ordered_and_sensible():
    m = tm.build_tire_model()
    assert m["_eras"] == [tm.ERA_2026]
    w = {c: m[c]["age_coef"] for c in tm.DRY_COMPOUNDS}
    assert w["SOFT"] >= w["MEDIUM"] - 1e-9 >= w["HARD"] - 2e-9 and w["HARD"] > 0
    o = m["_compound_offsets"]
    assert o["SOFT"] <= o["MEDIUM"] <= o["HARD"] == 0.0
    assert -0.08 < m["_fuel_coef"] < -0.03
    assert 13000 < m["_n_laps_fitted"] < 15000


def test_constrained_equals_free_when_data_already_ordered():
    l25 = tm.assign_stint_ids(tm.load_2025_staging_laps())
    clean = tm.clean_laps(l25)
    f = tm.fit_joint_model(clean)
    c = tm.fit_joint_model(clean, constrain=True)
    for k in ("SOFT", "MEDIUM"):
        assert c["offsets"][k] == pytest.approx(f["offsets"][k], abs=1e-6)
    for k in tm.DRY_COMPOUNDS:
        assert c["wear"][k][0] == pytest.approx(f["wear"][k][0], abs=1e-6)


def test_rain_and_red_flag_stints_excluded():
    laps = tm.assign_stint_ids(tm.load_all_laps())
    clean = tm.clean_laps(laps)
    assert not clean["rainfall"].astype(bool).any()
    assert len(tm.clean_laps(laps, drop_red_flag_stints=False)) > len(clean)
