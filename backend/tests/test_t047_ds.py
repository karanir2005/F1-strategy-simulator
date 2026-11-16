"""T-047: compound durability from stint lengths, softer-first tie-break, Austin/Mexico rebuilt from 2026."""
import pytest

from app.core import tire_model as tm
from app.core.optimizer import optimize
from app.core.track_params import get_track


def test_durability_default_orders_wear_and_keeps_level():
    on = tm.build_tire_model()
    off = tm.build_tire_model(durability=None)
    assert on["_durability_ratios"]["HARD"] == 1.0
    assert on["_durability_ratios"]["SOFT"] < on["_durability_ratios"]["MEDIUM"] < 1.0
    assert on["SOFT"]["age_coef"] > on["MEDIUM"]["age_coef"] > on["HARD"]["age_coef"]
    assert "_durability_ratios" not in off
    n = {c: 1.0 for c in tm.DRY_COMPOUNDS}
    assert sum(on[c]["age_coef"] for c in n) == pytest.approx(sum(off[c]["age_coef"] for c in n), rel=0.3)
    with pytest.raises(ValueError):
        tm.build_tire_model(durability="nope")


def test_equal_stints_are_listed_softer_first():
    m = tm.build_tire_model()
    t = get_track("austin")
    r = optimize(m, t["total_laps"], t["pit_loss_s"], t["base_laptime_s"], t["max_stint_laps"],
                 typical_stint_laps=t["typical_stint_laps"], n_sims=50, top_n=10, track_id="austin")
    order = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}
    for p in r.plans:
        cs = [s["compound"] for s in p["stints"]]
        if len(set(cs)) == 2 and p["n_stops"] == 1:
            assert order[cs[0]] < order[cs[1]], cs


def test_austin_and_mexico_are_2026_sourced():
    for tid in ("austin", "mexico"):
        t = get_track(tid)
        assert t["source_season"] == 2026 and t["already_raced_2026"], tid
    a = get_track("austin")
    assert a["compounds"] == {"HARD": "C2", "MEDIUM": "C3", "SOFT": "C4"}
    assert a["max_stint_laps"]["HARD"] > 28  # the old 2025 cap that forced H28/M28
    assert "red-flag" in get_track("mexico")["notes"]


def test_t050_wet_brazil_excluded_from_fit():
    import pandas as pd
    from app.core import tire_model as tm
    laps = tm.load_all_laps()
    assert laps["race_file"].str.contains("Brazil_2026").any()
    kept = tm.drop_wet_races(laps)
    assert not kept["race_file"].str.contains("Brazil_2026").any()
    assert kept["race_file"].str.contains("Canada_2026").any()  # 1.2% rain-tyre laps stays
