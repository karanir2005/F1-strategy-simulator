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
# pooled across the 10 races in pull_races.py's RACES list (369 stops,
# 10-100s sanity range to drop formation-lap/drive-through artifacts).
# Value is the median (23.55s), not the mean (25.35s), because Monaco's
# narrow pit lane produces a heavy right tail of traffic-queued stops
# (std 13.76s vs 1-6s elsewhere) that would otherwise skew a single
# global constant toward an atypical track. Re-run calibrate_pit_loss.py
# if the pulled race set changes materially.
DEFAULT_PIT_LOSS_SECONDS = 23.5


@dataclass
class Stint:
    compound: str
    laps: int


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

    strategy: ordered stints (compound + stint length in laps).
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
        for tire_age in range(1, stint.laps + 1):
            lap_number += 1
            tire_delta = predict_delta(tire_model, stint.compound, tire_age)
            fuel_delta = fuel_effect(tire_model, lap_number, total_laps)
            pace_offset = compound_pace_offset(tire_model, stint.compound)
            lap_time = base_laptime + tire_delta + fuel_delta + pace_offset

            if lap_time_noise_std:
                lap_time += rng.normal(0.0, lap_time_noise_std)

            # pit loss is charged to the out-lap (first lap of every
            # stint after the first) -- a common simplification rather
            # than splitting it across the in-lap and out-lap
            if tire_age == 1 and stint_index > 0:
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
