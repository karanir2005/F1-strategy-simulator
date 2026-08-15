"""
Historical replay validation: reconstruct a real driver's actual pit
strategy from pulled race data, simulate it, and compare simulated total
race time against what actually happened.

Run this from project root with: python validate_replay.py
"""

import sys

sys.path.insert(0, "backend")

import pandas as pd

from app.core.race_simulator import Stint, simulate_race
from app.core.tire_model import build_tire_model


def reconstruct_strategy(driver_laps: pd.DataFrame) -> list[Stint]:
    """Rebuild the stint-by-stint strategy a driver actually ran from their lap data."""
    driver_laps = driver_laps.sort_values("lap_number")
    is_new_stint = driver_laps["tire_age"] == 1
    stint_id = is_new_stint.cumsum()

    strategy = []
    for _, stint_laps in driver_laps.groupby(stint_id):
        compound = stint_laps["compound"].iloc[0]
        strategy.append(Stint(compound=compound, laps=len(stint_laps)))
    return strategy


def validate(race_csv: str, driver: str, tire_model: dict) -> None:
    laps = pd.read_csv(race_csv)
    driver_laps = laps[laps["driver"] == driver]

    strategy = reconstruct_strategy(driver_laps)
    actual_total = driver_laps["lap_time_seconds"].sum()
    base_laptime = driver_laps["lap_time_seconds"].min()  # fastest lap as fresh-tire proxy

    result = simulate_race(strategy, base_laptime=base_laptime, tire_model=tire_model)

    error = result.total_time - actual_total
    pct_error = 100 * error / actual_total

    print(f"{race_csv} - {driver}")
    print(f"  strategy: {[(s.compound, s.laps) for s in strategy]}")
    print(f"  actual total:    {actual_total:8.1f}s")
    print(f"  simulated total: {result.total_time:8.1f}s")
    print(f"  error:           {error:+8.1f}s ({pct_error:+.2f}%)")
    print()


if __name__ == "__main__":
    tire_model = build_tire_model()

    checks = [
        ("data/processed/China_2026_laps.csv", "GAS"),
        ("data/processed/Britain_2026_laps.csv", "GAS"),
        ("data/processed/Austria_2026_laps.csv", "GAS"),
    ]
    for race_csv, driver in checks:
        validate(race_csv, driver, tire_model)
