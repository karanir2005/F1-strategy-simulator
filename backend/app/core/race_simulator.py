"""
Race simulator: deterministic single-strategy lap-by-lap simulation.

Given a pit strategy (ordered stints of compound + stint length) and a
circuit's base lap time, simulates total race time using tire_model's
degradation curves plus a pit stop time loss. No randomness here --
Monte Carlo noise (variable degradation, variable pit duration, safety
cars) is layered on top in monte_carlo.py, which is the next module.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

# make `app.core.*` importable regardless of the caller's cwd (mirrors the
# sys.path.insert(0, "backend") convention used by test_data_load.py, but
# resolved from this file's location instead of assuming cwd == repo root)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.core.tire_model import compound_pace_offset, fuel_effect, predict_delta

# Calibrated from real pit-lane duration data (data_loader.py's
# extract_pit_stops, PitOutTime - PitInTime) via calibrate_pit_loss.py,
# pooled across the original 10 races (369 stops, 10-100s sanity range to
# drop formation-lap/drive-through artifacts): median 23.55s, mean 25.35s.
# T-028 (2026-10-23) re-ran it over all 16 races in data/processed (294 clean
# tyre-change stops, Monaco excluded, not under SC/VSC/red): median 22.88s,
# -2.6%, inside the 10% re-tune threshold so the constant is left alone.
# Value is the median, not the mean, because Monaco's
# narrow pit lane produces a heavy right tail of traffic-queued stops
# (std 13.76s vs 1-6s elsewhere) that would otherwise skew a single
# global constant toward an atypical track. Re-run calibrate_pit_loss.py
# if the pulled race set changes materially.
DEFAULT_PIT_LOSS_SECONDS = 23.5


@dataclass
class Stint:
    """
    One stint of a strategy.

    start_age: tire age (in laps) on the first lap of this stint. 1 means a
        new set (the default, and the only option before T-004). >1 models
        a used set -- e.g. one scrubbed in qualifying or saved from an
        earlier stint -- so the stint runs tire ages start_age ..
        start_age + laps - 1 and pays for the wear already on it. About
        29% of real post-pit stints in the pulled data start on a used set
        (see team/ROADMAP.md), so this is not an edge case.
    """

    compound: str
    laps: int
    start_age: int = 1

    def __post_init__(self):
        if self.start_age < 1:
            raise ValueError(f"start_age must be >= 1, got {self.start_age}")


@dataclass
class SimulationResult:
    total_time: float
    lap_times: List[float] = field(default_factory=list)
    pit_laps: List[int] = field(default_factory=list)


def simulate_race(
    strategy: List[Stint],
    base_laptime: float,
    tire_model: dict,
    pit_loss_seconds: float = DEFAULT_PIT_LOSS_SECONDS,
    lap_time_noise_std: float = 0.0,
    pit_loss_noise_std: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> SimulationResult:
    """
    Simulate one race given a pit strategy.

    strategy: ordered stints (compound + stint length in laps, plus an
        optional start_age for used sets -- see Stint).
    base_laptime: fresh-tire, low-fuel reference lap time for the circuit
        being simulated (e.g. a driver's fastest lap of the race) --
        circuit-specific, so it's passed in rather than assumed. Treated
        as the HARD-compound baseline; compound_pace_offset() shifts
        SOFT/MEDIUM laps relative to it. Paired with fuel_effect(), which
        is anchored to 0 at the final lap, so base_laptime represents a
        fully light car.
    lap_time_noise_std / pit_loss_noise_std: standard deviation (seconds)
        of Gaussian noise added per lap / per pit stop. Both default to 0
        (fully deterministic, matching prior behavior) -- monte_carlo.py
        is what turns these on to build a distribution of outcomes.
    rng: shared numpy Generator for reproducible Monte Carlo runs. Only
        created locally (unseeded) if noise is requested and no rng is
        passed in.
    """
    if rng is None and (lap_time_noise_std or pit_loss_noise_std):
        rng = np.random.default_rng()

    total_laps = sum(stint.laps for stint in strategy)
    lap_times = []
    pit_laps = []
    lap_number = 0

    for stint_index, stint in enumerate(strategy):
        for lap_in_stint in range(stint.laps):
            lap_number += 1
            # predict_delta is relative to age 2, so a used set (start_age
            # > 1) is simply evaluated further along the same curve -- no
            # separate "used tyre" model, which assumes a set's wear only
            # depends on laps already run on it, not how they were run
            tire_age = stint.start_age + lap_in_stint
            tire_delta = predict_delta(tire_model, stint.compound, tire_age)
            fuel_delta = fuel_effect(tire_model, lap_number, total_laps)
            pace_offset = compound_pace_offset(tire_model, stint.compound)
            lap_time = base_laptime + tire_delta + fuel_delta + pace_offset

            if lap_time_noise_std:
                lap_time += rng.normal(0.0, lap_time_noise_std)

            # pit loss is charged to the out-lap (first lap of every
            # stint after the first) -- a common simplification rather
            # than splitting it across the in-lap and out-lap. Keyed on
            # position within the stint, not tire_age == 1, so a stop onto
            # a used set is still charged
            if lap_in_stint == 0 and stint_index > 0:
                pit_loss = pit_loss_seconds
                if pit_loss_noise_std:
                    pit_loss += rng.normal(0.0, pit_loss_noise_std)
                lap_time += pit_loss
                pit_laps.append(lap_number)

            lap_times.append(lap_time)

    return SimulationResult(
        total_time=sum(lap_times), lap_times=lap_times, pit_laps=pit_laps
    )


if __name__ == "__main__":
    from app.core.tire_model import build_tire_model

    tire_model = build_tire_model()

    one_stop = [Stint("MEDIUM", 25), Stint("HARD", 32)]
    two_stop = [Stint("SOFT", 15), Stint("MEDIUM", 20), Stint("HARD", 22)]

    for name, strategy in [("1-stop", one_stop), ("2-stop", two_stop)]:
        result = simulate_race(strategy, base_laptime=90.0, tire_model=tire_model)
        print(f"{name}: total={result.total_time:.1f}s pit_laps={result.pit_laps}")
