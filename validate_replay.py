"""
Historical replay validation: reconstruct a real driver's actual pit
strategy from pulled race data, simulate it, and compare simulated total
race time against what actually happened.

Run this from project root with: python validate_replay.py

T-002 (2026-10-19): stints now come from data_loader's tyre_stint (real
tyre changes only -- same-tyre pit-lane passes no longer look like stops,
and stops onto used sets no longer get merged into the previous stint),
and each Stint carries start_age = the real tyre age on its first lap.
The actual total is now taken from the timing line (lap_end_s), because
laps without a LapTime (mostly SC laps) are kept in the CSV and summing
lap_time_seconds would silently drop them.

Known limitation (fixed by T-003's scorecard, not here): base_laptime is
the driver's own fastest lap of the race being validated, so this check
leaks the answer and is too coarse to rank strategies.
"""

import sys

sys.path.insert(0, "backend")

import pandas as pd

from app.core.race_simulator import Stint, simulate_race
from app.core.tire_model import build_tire_model


def reconstruct_strategy(driver_laps: pd.DataFrame) -> list[Stint]:
    """
    Rebuild the stint-by-stint strategy a driver actually ran from their
    lap data, including each stint's start_age (tyre age on its first
    lap; > 1 for a used set).

    T-002 CSVs: one stint per tyre_stint (FastF1 Stint split into real
    tyre changes vs same-tyre pit-lane passes, see data_loader.py).
    Legacy CSVs (no tyre_stint column): falls back to "new stint when
    tire_age == 1", which merges stops onto used sets into the previous
    stint -- re-extract with pull_races.py.
    """
    driver_laps = driver_laps.sort_values("lap_number")
    if "tyre_stint" in driver_laps.columns:
        stint_key = driver_laps["tyre_stint"]
    else:
        stint_key = (driver_laps["tire_age"] == 1).cumsum()

    strategy = []
    for _, stint_laps in driver_laps.groupby(stint_key, sort=True):
        compound = stint_laps["compound"].iloc[0]
        if pd.isna(compound):
            raise ValueError(
                f"stint starting lap {int(stint_laps['lap_number'].iloc[0])} has no "
                "tyre data (compound unknown) -- can't replay it"
            )
        if "stint_start_age" in stint_laps.columns:
            first_age = stint_laps["stint_start_age"].iloc[0]
        else:
            first_age = stint_laps["tire_age"].iloc[0]
        start_age = int(first_age) if pd.notna(first_age) and first_age >= 1 else 1
        strategy.append(Stint(compound=compound, laps=len(stint_laps), start_age=start_age))
    return strategy


def actual_race_time(driver_laps: pd.DataFrame) -> float:
    """
    Real elapsed time over the driver's laps. With T-002 CSVs this is
    lap_end_s(last) - lap_end_s(first) + first lap's time, which stays
    correct when some laps have no LapTime. (Lap 1's LapStartTime is off by
    up to ~0.4 s, so lap 1 uses its LapTime when available.) Legacy CSVs:
    sum of lap times.
    """
    driver_laps = driver_laps.sort_values("lap_number")
    if "lap_end_s" not in driver_laps.columns:
        return float(driver_laps["lap_time_seconds"].sum())
    first = driver_laps.iloc[0]
    first_lap_time = first["lap_time_seconds"]
    if pd.isna(first_lap_time):
        first_lap_time = first["lap_end_s"] - first["lap_start_s"]
    return float(driver_laps["lap_end_s"].iloc[-1] - first["lap_end_s"] + first_lap_time)


def validate(race_csv: str, driver: str, tire_model: dict) -> dict:
    laps = pd.read_csv(race_csv)
    driver_laps = laps[laps["driver"] == driver]

    strategy = reconstruct_strategy(driver_laps)
    actual_total = actual_race_time(driver_laps)
    base_laptime = driver_laps["lap_time_seconds"].min()  # fastest lap as fresh-tire proxy

    result = simulate_race(strategy, base_laptime=base_laptime, tire_model=tire_model)

    error = result.total_time - actual_total
    pct_error = 100 * error / actual_total

    print(f"{race_csv} - {driver}")
    print(f"  strategy: {[(s.compound, s.laps, s.start_age) for s in strategy]}  (compound, laps, start_age)")
    print(f"  actual total:    {actual_total:8.1f}s")
    print(f"  simulated total: {result.total_time:8.1f}s")
    print(f"  error:           {error:+8.1f}s ({pct_error:+.2f}%)")

    out = {"error_s": error, "pct_error": pct_error, "strategy": strategy}
    # The engine has no safety cars yet, so SC/VSC/red laps are a known,
    # one-signed miss. Report the error on the remaining laps separately
    # so tyre/fuel-model changes aren't drowned out by SC luck.
    if "is_neutralised" in driver_laps.columns:
        ordered = driver_laps.sort_values("lap_number").reset_index(drop=True)
        racing = ~ordered["is_neutralised"].astype(bool) & ordered["lap_time_seconds"].notna()
        sim_laps = pd.Series(result.lap_times[: len(ordered)])
        act_racing = ordered.loc[racing, "lap_time_seconds"].sum()
        sim_racing = sim_laps[racing].sum()
        err_racing = sim_racing - act_racing
        print(
            f"  excl. {int((~racing).sum())} SC/VSC/red laps: actual {act_racing:8.1f}s, "
            f"sim {sim_racing:8.1f}s, error {err_racing:+.1f}s ({100 * err_racing / act_racing:+.2f}%)"
        )
        out.update(error_racing_s=err_racing, pct_error_racing=100 * err_racing / act_racing)
    print()
    return out


if __name__ == "__main__":
    tire_model = build_tire_model()

    checks = [
        ("data/processed/China_2026_laps.csv", "GAS"),
        ("data/processed/Britain_2026_laps.csv", "GAS"),
        ("data/processed/Austria_2026_laps.csv", "GAS"),
        # T-002: replays that exercise stops onto USED sets (start_age > 1),
        # which the pre-T-002 stint rule merged into the previous stint
        ("data/processed/Britain_2026_laps.csv", "PIA"),  # MEDIUM age 4 (L37), SOFT age 4 (L48)
        ("data/processed/Hungary_2026_laps.csv", "VER"),  # SOFT age 4 (L42)
    ]
    for race_csv, driver in checks:
        validate(race_csv, driver, tire_model)
