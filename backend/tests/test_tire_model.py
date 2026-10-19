import numpy as np
import pandas as pd
import pytest

from app.core.tire_model import (
    assign_stint_ids,
    build_tire_model,
    clean_laps,
    compound_pace_offset,
    fit_joint_model,
    fuel_effect,
    load_all_laps,
    predict_delta,
)


def test_predict_delta_is_zero_at_reference_age(toy_tire_model):
    # delta is defined relative to age 2 -- it must be exactly 0 there
    assert predict_delta(toy_tire_model, "MEDIUM", 2) == pytest.approx(0.0)


def test_predict_delta_matches_hand_computed_linear_curve(toy_tire_model):
    # MEDIUM: age_coef=0.05, age2_coef=0 -> delta(age) = 0.05 * (age - 2)
    assert predict_delta(toy_tire_model, "MEDIUM", 10) == pytest.approx(0.05 * (10 - 2))


def test_predict_delta_extrapolates_linearly_past_cliff(toy_tire_model):
    # SOFT's cliff_age (10) is below its max_reliable_age (20), so 10 is
    # the boundary; beyond it, delta should grow at exactly
    # cliff_penalty_per_lap per lap on top of the boundary's value
    boundary_delta = predict_delta(toy_tire_model, "SOFT", 10)
    delta_at_15 = predict_delta(toy_tire_model, "SOFT", 15, cliff_penalty_per_lap=0.3)
    assert delta_at_15 == pytest.approx(boundary_delta + 0.3 * 5)


def test_predict_delta_extrapolates_past_max_reliable_age_when_no_cliff(toy_tire_model):
    # HARD has no detected cliff, so max_reliable_age (40) is the boundary
    boundary_delta = predict_delta(toy_tire_model, "HARD", 40)
    delta_at_50 = predict_delta(toy_tire_model, "HARD", 50, cliff_penalty_per_lap=0.3)
    assert delta_at_50 == pytest.approx(boundary_delta + 0.3 * 10)


def test_fuel_effect_is_zero_at_final_lap(toy_tire_model):
    assert fuel_effect(toy_tire_model, lap_number=50, total_laps=50) == pytest.approx(0.0)


def test_fuel_effect_is_positive_early_in_the_race(toy_tire_model):
    # negative fuel_coef => a heavier, early-race car is slower (positive
    # delta) relative to the light-fuel final lap
    assert fuel_effect(toy_tire_model, lap_number=1, total_laps=50) > 0


def test_fuel_effect_scales_linearly_with_laps_remaining(toy_tire_model):
    # 50 laps remaining vs. 25 laps remaining -- an exact doubling
    delta_50_remaining = fuel_effect(toy_tire_model, lap_number=0, total_laps=50)
    delta_25_remaining = fuel_effect(toy_tire_model, lap_number=25, total_laps=50)
    assert delta_50_remaining == pytest.approx(2 * delta_25_remaining)


def test_build_tire_model_returns_all_compounds_and_fuel_coef():
    """Integration test against the real data in data/processed/ -- skipped implicitly if that data is missing (raises FileNotFoundError)."""
    model = build_tire_model()
    for compound in ("SOFT", "MEDIUM", "HARD"):
        assert compound in model
        assert "age_coef" in model[compound]
        assert "age2_coef" in model[compound]
        assert "max_reliable_age" in model[compound]
    assert isinstance(model["_fuel_coef"], float)


def test_hard_has_zero_compound_offset(toy_tire_model):
    # HARD is the reference compound -- everything else is measured relative to it
    assert compound_pace_offset(toy_tire_model, "HARD") == pytest.approx(0.0)


def test_softer_compounds_are_faster_than_harder_ones(toy_tire_model):
    """
    This is the property a user found broken in the frontend: comparing
    SOFT/MEDIUM/HARD single-stint no-pit strategies at the same lap count
    showed nearly identical lines, because predict_delta() is always 0 at
    each compound's own reference age -- it carries no information that a
    fresh SOFT tyre is inherently faster than a fresh HARD one. That's
    what compound_pace_offset() exists to supply.
    """
    soft = compound_pace_offset(toy_tire_model, "SOFT")
    medium = compound_pace_offset(toy_tire_model, "MEDIUM")
    hard = compound_pace_offset(toy_tire_model, "HARD")
    assert soft < medium < hard


def test_build_tire_model_includes_compound_offsets():
    model = build_tire_model()
    assert set(model["_compound_offsets"].keys()) == {"SOFT", "MEDIUM", "HARD"}
    assert model["_compound_offsets"]["HARD"] == pytest.approx(0.0)


def test_predict_delta_clamps_a_dipping_curve_to_be_monotonic():
    """
    A user found this exact bug: fitted curves can dip below zero for a
    stretch (fuel/track-evolution effects not fully separated from wear
    during fitting -- see this module's top docstring), which predicts a
    worn tire is FASTER than a fresh one. That's not physically possible,
    so predict_delta() must clamp it. Build a model whose raw quadratic
    genuinely dips (age_coef < 0, small positive age2_coef, vertex well
    inside the domain) and confirm the dip is gone.
    """
    dipping_model = {
        "SOFT": {
            "age_coef": -0.5,
            "age2_coef": 0.01,  # vertex at age = 0.5/(2*0.01) = 25
            "cliff_age": None,
            "max_reliable_age": 40,
        },
        "_cliff_penalties": {"SOFT": 0.3},
    }
    deltas = [predict_delta(dipping_model, "SOFT", age) for age in range(2, 41)]
    # raw curve would dip well below 0 around age 25 -- clamped version
    # must never decrease and must never go negative
    assert min(deltas) == pytest.approx(0.0)
    assert all(b >= a - 1e-9 for a, b in zip(deltas, deltas[1:]))


def test_predict_delta_uses_compound_specific_cliff_penalty():
    model_with_penalties = {
        "SOFT": {"age_coef": 0.0, "age2_coef": 0.0, "cliff_age": None, "max_reliable_age": 10},
        "HARD": {"age_coef": 0.0, "age2_coef": 0.0, "cliff_age": None, "max_reliable_age": 10},
        "_cliff_penalties": {"SOFT": 1.0, "HARD": 0.2},
    }
    # 5 laps past boundary at each compound's own penalty rate
    assert predict_delta(model_with_penalties, "SOFT", 15) == pytest.approx(1.0 * 5)
    assert predict_delta(model_with_penalties, "HARD", 15) == pytest.approx(0.2 * 5)


def test_predict_delta_falls_back_to_default_cliff_penalty_when_unset(toy_tire_model):
    # toy_tire_model has no "_cliff_penalties" key -- must not KeyError
    assert "_cliff_penalties" not in toy_tire_model
    assert predict_delta(toy_tire_model, "HARD", 50) > 0


def test_build_tire_model_includes_cliff_penalties():
    model = build_tire_model()
    assert set(model["_cliff_penalties"].keys()) == {"SOFT", "MEDIUM", "HARD"}
    # ordering reflects real-world severity: softer compounds fall off harder
    assert (
        model["_cliff_penalties"]["SOFT"]
        > model["_cliff_penalties"]["MEDIUM"]
        > model["_cliff_penalties"]["HARD"]
    )


# ---------------------------------------------------------------------------
# T-002: stints, cleaning, joint estimator
# ---------------------------------------------------------------------------


def _t002_laps(**overrides) -> pd.DataFrame:
    """Two-stint driver in the T-002 CSV schema: MEDIUM new L1-4, SOFT used (age 5) L5-8."""
    df = pd.DataFrame(
        {
            "race_file": "X_2026_laps.csv",
            "driver": "AAA",
            "lap_number": range(1, 9),
            "lap_time_seconds": [95.0, 90.0, 90.1, 90.6, 112.0, 89.5, 89.6, 89.7],
            "compound": ["MEDIUM"] * 4 + ["SOFT"] * 4,
            "tire_age": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "tyre_stint": [1] * 4 + [2] * 4,
            "pit_in_lap": [0, 0, 0, 1, 0, 0, 0, 0],
            "pit_out_lap": [0, 0, 0, 0, 1, 0, 0, 0],
            "is_green_flag": [True] * 8,
            "is_neutralised": [False] * 8,
            "tyre_data_missing": [False] * 8,
        }
    )
    for col, values in overrides.items():
        df[col] = values
    return df


def test_assign_stint_ids_uses_real_tyre_stints():
    # the used SOFT set (age 5) must be its own stint -- the old
    # tire_age == 1 rule merged it into the MEDIUM stint
    laps = assign_stint_ids(_t002_laps())
    assert laps["stint_id"].tolist() == [1, 1, 1, 1, 2, 2, 2, 2]


def test_assign_stint_ids_legacy_fallback_without_tyre_stint():
    legacy = _t002_laps().drop(columns=["tyre_stint"])
    laps = assign_stint_ids(legacy)
    assert laps["stint_id"].tolist() == [1] * 8  # documented legacy limitation


def test_clean_laps_drops_used_set_out_lap_lap1_and_in_lap():
    clean = clean_laps(_t002_laps())
    # lap 1 (start), lap 4 (in-lap), lap 5 (out-lap onto a USED set, age 5)
    assert clean["lap_number"].tolist() == [2, 3, 6, 7, 8]


def test_clean_laps_drops_restart_yellow_nan_and_unknown_tyre_laps():
    laps = _t002_laps(
        is_neutralised=[False, False, True, False, False, False, False, False],
        is_green_flag=[True, True, False, True, True, True, False, True],  # L7 local yellow
        lap_time_seconds=[95.0, 90.0, 120.0, 90.6, 112.0, 89.5, 89.6, np.nan],
        pit_in_lap=[0] * 8,
        pit_out_lap=[0] * 8,
        tyre_data_missing=[False, False, False, False, False, True, False, False],
    )
    clean = clean_laps(laps)
    # L1 start, L3 SC, L4 restart (previous lap neutralised), L6 unknown
    # tyre, L7 yellow, L8 no lap time
    assert clean["lap_number"].tolist() == [2, 5]


def _synthetic_joint_laps(rng_seed=0, noise=0.0) -> pd.DataFrame:
    """
    Two races x six drivers, two stints each with the stop on a different
    lap per driver, compounds rotated. Lap time = driver-race pace +
    race-specific fuel slope * lap + compound offset + linear wear * age.
    """
    rng = np.random.default_rng(rng_seed)
    fuel = {"A_laps.csv": -0.06, "B_laps.csv": -0.03}
    offsets = {"SOFT": -0.6, "MEDIUM": -0.25, "HARD": 0.0}
    wear = {"SOFT": 0.09, "MEDIUM": 0.05, "HARD": 0.02}
    plans = [("SOFT", "HARD"), ("MEDIUM", "HARD"), ("HARD", "MEDIUM"),
             ("SOFT", "MEDIUM"), ("MEDIUM", "SOFT"), ("HARD", "SOFT")]
    rows = []
    for race, f in fuel.items():
        for d, (c1, c2) in enumerate(plans):
            stop = 12 + 3 * d
            pace = 80.0 + d
            for lap in range(1, 41):
                c, age = (c1, lap) if lap <= stop else (c2, lap - stop)
                t = pace + f * lap + offsets[c] + wear[c] * age + rng.normal(0, noise)
                rows.append({"race_file": race, "driver": f"D{d}", "lap_number": lap,
                             "lap_time_seconds": t, "compound": c, "tire_age": float(age)})
    return pd.DataFrame(rows), fuel, offsets, wear


def test_fit_joint_model_recovers_known_parameters():
    laps, fuel, offsets, wear = _synthetic_joint_laps()
    fit = fit_joint_model(laps)
    for race, slope in fuel.items():
        assert fit["fuel_by_race"][race] == pytest.approx(slope, abs=1e-9)
    assert fit["fuel_coef"] == pytest.approx(np.mean(list(fuel.values())), abs=1e-9)
    for compound in ("SOFT", "MEDIUM", "HARD"):
        assert fit["offsets"][compound] == pytest.approx(offsets[compound], abs=1e-9)
        assert fit["wear"][compound][0] == pytest.approx(wear[compound], abs=1e-9)
        assert fit["wear"][compound][1] == 0.0
    assert np.abs(fit["residual"]).max() < 1e-8


def test_build_tire_model_accepts_preloaded_synthetic_laps():
    laps, _, _, wear = _synthetic_joint_laps(noise=0.05)
    # legacy-schema frame (no T-002 columns): must still fit
    laps = laps.assign(pit_in_lap=0, is_green_flag=True)
    model = build_tire_model(laps=laps)
    for compound in ("SOFT", "MEDIUM", "HARD"):
        assert model[compound]["age_coef"] == pytest.approx(wear[compound], abs=0.01)
    assert model["_fuel_coef"] == pytest.approx(-0.045, abs=0.01)


def test_real_soft_curve_increases_with_age():
    """
    T-004 QA N-3: the pre-T-002 fit gave SOFT a flat 0.00 delta from age 2
    to 37, so a used SOFT set simulated as fast as a new one. Integration
    test against data/processed/.
    """
    model = build_tire_model()
    soft = [predict_delta(model, "SOFT", age) for age in (2, 5, 10, 20)]
    assert all(b > a for a, b in zip(soft, soft[1:]))
    for compound in ("SOFT", "MEDIUM", "HARD"):
        assert model[compound]["age_coef"] > 0


def test_real_data_has_no_merged_stints():
    """Every T-002 tyre stint is one compound with tyre age stepping by exactly +1."""
    laps = assign_stint_ids(load_all_laps())
    known = laps[laps["compound"].notna() & laps["tire_age"].notna()]
    grouped = known.groupby(["race_file", "driver", "stint_id"])
    assert (grouped["compound"].nunique() == 1).all()
    steps = grouped["tire_age"].diff().dropna()
    assert (steps == 1).all()


def test_build_tire_model_exposes_fit_diagnostics():
    model = build_tire_model()
    assert set(model["_compound_offsets_fitted"]) == {"SOFT", "MEDIUM", "HARD"}
    assert len(model["_fuel_coef_by_race"]) >= 1
    assert model["_n_laps_fitted"] > 0
