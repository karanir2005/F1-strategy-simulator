import numpy as np
import pytest

from app.core.race_simulator import Stint, simulate_race


def test_compound_pace_offset_shifts_lap_time(toy_tire_model):
    """
    Isolate the compound pace offset from degradation: use a model where
    every compound has an IDENTICAL (flat) degradation curve, so any
    difference between two single-lap runs must come purely from
    compound_pace_offset -- this is the "SOFT should just be faster"
    property a real user found missing in the frontend.
    """
    flat_model = {
        "SOFT": {"age_coef": 0.0, "age2_coef": 0.0, "cliff_age": None, "max_reliable_age": 40},
        "MEDIUM": {"age_coef": 0.0, "age2_coef": 0.0, "cliff_age": None, "max_reliable_age": 40},
        "HARD": {"age_coef": 0.0, "age2_coef": 0.0, "cliff_age": None, "max_reliable_age": 40},
        "_fuel_coef": 0.0,
        "_compound_offsets": {"SOFT": -0.5, "MEDIUM": -0.2, "HARD": 0.0},
    }
    soft = simulate_race([Stint("SOFT", 1)], base_laptime=90.0, tire_model=flat_model)
    medium = simulate_race([Stint("MEDIUM", 1)], base_laptime=90.0, tire_model=flat_model)
    hard = simulate_race([Stint("HARD", 1)], base_laptime=90.0, tire_model=flat_model)

    assert soft.lap_times[0] == pytest.approx(89.5)
    assert medium.lap_times[0] == pytest.approx(89.8)
    assert hard.lap_times[0] == pytest.approx(90.0)


def test_single_stint_has_no_pit_stops(toy_tire_model):
    strategy = [Stint("MEDIUM", 10)]
    result = simulate_race(strategy, base_laptime=90.0, tire_model=toy_tire_model)
    assert result.pit_laps == []
    assert len(result.lap_times) == 10


def test_pit_loss_charged_on_the_correct_lap(toy_tire_model):
    strategy = [Stint("MEDIUM", 5), Stint("HARD", 5)]
    result = simulate_race(strategy, base_laptime=90.0, tire_model=toy_tire_model)
    assert result.pit_laps == [6]  # first lap of the second stint


def test_pit_loss_seconds_is_added_exactly(toy_tire_model):
    strategy = [Stint("MEDIUM", 5), Stint("HARD", 5)]
    with_pit = simulate_race(
        strategy, base_laptime=90.0, tire_model=toy_tire_model, pit_loss_seconds=22.0
    )
    without_pit = simulate_race(
        strategy, base_laptime=90.0, tire_model=toy_tire_model, pit_loss_seconds=0.0
    )
    pit_lap_index = with_pit.pit_laps[0] - 1
    assert with_pit.lap_times[pit_lap_index] - without_pit.lap_times[
        pit_lap_index
    ] == pytest.approx(22.0)
    # every other lap should be unaffected by pit_loss_seconds
    for i in range(len(with_pit.lap_times)):
        if i != pit_lap_index:
            assert with_pit.lap_times[i] == pytest.approx(without_pit.lap_times[i])


def test_total_time_is_the_sum_of_lap_times(toy_tire_model):
    strategy = [Stint("SOFT", 8), Stint("MEDIUM", 12)]
    result = simulate_race(strategy, base_laptime=88.0, tire_model=toy_tire_model)
    assert len(result.lap_times) == 20
    assert result.total_time == pytest.approx(sum(result.lap_times))


def test_zero_noise_is_fully_deterministic(toy_tire_model):
    strategy = [Stint("MEDIUM", 15)]
    r1 = simulate_race(strategy, base_laptime=90.0, tire_model=toy_tire_model)
    r2 = simulate_race(strategy, base_laptime=90.0, tire_model=toy_tire_model)
    assert r1.total_time == r2.total_time


def test_noise_is_reproducible_given_identically_seeded_rngs(toy_tire_model):
    strategy = [Stint("MEDIUM", 15)]
    r1 = simulate_race(
        strategy,
        base_laptime=90.0,
        tire_model=toy_tire_model,
        lap_time_noise_std=0.5,
        rng=np.random.default_rng(123),
    )
    r2 = simulate_race(
        strategy,
        base_laptime=90.0,
        tire_model=toy_tire_model,
        lap_time_noise_std=0.5,
        rng=np.random.default_rng(123),
    )
    assert r1.total_time == pytest.approx(r2.total_time)


def test_noise_actually_perturbs_the_result(toy_tire_model):
    strategy = [Stint("MEDIUM", 15)]
    deterministic = simulate_race(strategy, base_laptime=90.0, tire_model=toy_tire_model)
    noisy = simulate_race(
        strategy,
        base_laptime=90.0,
        tire_model=toy_tire_model,
        lap_time_noise_std=5.0,
        rng=np.random.default_rng(1),
    )
    assert noisy.total_time != pytest.approx(deterministic.total_time)
