"""T-043: pace-ordered compound offsets and cross-era track wear transfer (tire_model)."""
import pytest

from app.core import tire_model as tm


def test_project_pace_ordering():
    # inverted 2026-style free fit -> both clipped to HARD
    assert tm.project_pace_ordering({"SOFT": 0.13, "MEDIUM": 0.19, "HARD": 0.0}) == {"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0}
    # textbook ordering untouched
    assert tm.project_pace_ordering({"SOFT": -0.37, "MEDIUM": -0.05, "HARD": 0.0}) == {"SOFT": -0.37, "MEDIUM": -0.05, "HARD": 0.0}
    # SOFT slower than a faster MEDIUM -> SOFT pulled up to MEDIUM
    assert tm.project_pace_ordering({"SOFT": 0.1, "MEDIUM": -0.2, "HARD": 0.0})["SOFT"] == -0.2


def test_default_offset_mode_is_pace_ordered_and_keeps_wear_ordering():
    m = tm.build_tire_model()
    assert m["_offset_mode"] == tm.DEFAULT_OFFSET_MODE == "pace_ordered"
    off = m["_compound_offsets"]
    assert off["SOFT"] <= off["MEDIUM"] <= off["HARD"] == 0.0
    assert m["SOFT"]["age_coef"] >= m["MEDIUM"]["age_coef"] - 1e-9 >= m["HARD"]["age_coef"] - 2e-9 >= -2e-9


def test_joint_mode_still_reproduces_t020_fit():
    m = tm.build_tire_model(offset_mode="joint")
    assert m["_compound_offsets"]["SOFT"] == pytest.approx(-0.123, abs=0.01)
    with pytest.raises(ValueError):
        tm.build_tire_model(offset_mode="nope")


def test_transfer_track_wear_scales_and_keeps_order():
    m = tm.build_tire_model()
    other = tm.build_tire_model(laps=tm.load_all_laps(eras=(tm.ERA_2025,)), eras=(tm.ERA_2025,))
    tid = "austria"
    base = dict(m)
    base["_track_wear"] = {k: v for k, v in m["_track_wear"].items() if k != tid}  # pretend not raced
    out, lam = tm.transfer_track_wear(base, other, tid)
    assert 0.0 <= lam <= 1.0
    v = out["_track_wear"][tid]
    assert v["SOFT"]["age_coef"] >= v["MEDIUM"]["age_coef"] >= v["HARD"]["age_coef"] >= 0
    same, none = tm.transfer_track_wear(base, other, "no_such_track")
    assert none is None and same is base
