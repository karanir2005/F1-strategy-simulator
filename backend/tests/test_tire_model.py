import pytest

from app.core.tire_model import (
    build_tire_model,
    compound_pace_offset,
    fuel_effect,
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
