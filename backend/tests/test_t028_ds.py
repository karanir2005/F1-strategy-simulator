"""T-028: per-track wear (tire_model.fit_track_wear / model_for_track)."""
import pytest

from app.core import tire_model as tm


@pytest.fixture(scope="module")
def model():
    return tm.build_tire_model()


def test_track_ids_map_to_track_params():
    assert tm.track_id_of_race("x/LasVegas_2025_laps.csv") == "las_vegas"
    assert tm.track_id_of_race("AbuDhabi_2026_laps.csv") == "abudhabi"
    assert tm.track_id_of_race("Australia_2026_laps.csv") == "australia"


def test_track_wear_keeps_ordering_and_pins_sparse_compounds(model):
    tw = model["_track_wear"]
    assert len(tw) == 16
    for tid, v in tw.items():
        s, m, h = (v[c]["age_coef"] for c in ("SOFT", "MEDIUM", "HARD"))
        assert s >= m - 1e-9 >= h - 2e-9 >= -2e-9, tid
        for c in v.values():
            if c["n_stints"] < tm.MIN_TRACK_STINTS:
                assert c["pooled"] and abs(c["age_coef"] - c["global_age_coef"]) < 1e-6


def test_model_for_track_is_a_copy_and_noop_without_track(model):
    assert tm.model_for_track(model, None) is model
    assert tm.model_for_track(model, "no_such_track") is model
    g = model["HARD"]["age_coef"]
    t = tm.model_for_track(model, "barcelona")
    assert t is not model and model["HARD"]["age_coef"] == g
    assert t["HARD"]["age_coef"] != g
    assert t["_fuel_coef"] == model["_fuel_coef"] and t["SOFT"]["max_reliable_age"] == model["SOFT"]["max_reliable_age"]
