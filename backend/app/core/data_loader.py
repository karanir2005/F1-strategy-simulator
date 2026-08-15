"""
Data loader module for FastF1 race session data.
Extracts and cleans lap-by-lap metrics for simulation.
"""

import fastf1
import pandas as pd
from typing import Tuple, Optional


def load_race_session(year: int, gp: str, session_type: str = "R") -> fastf1.core.Session:
    """
    Load a FastF1 race session and cache the data.
    
    Args:
        year: Race year (e.g., 2024)
        gp: Grand Prix name (e.g., 'Bahrain')
        session_type: 'R' for Race, 'Q' for Qualifying
    
    Returns:
        Loaded FastF1 Session object
    """
    session = fastf1.get_session(year, gp, session_type)
    session.load()
    return session


def extract_laps_df(session: fastf1.core.Session) -> pd.DataFrame:
    """
    Extract clean lap-by-lap data from a race session.
    
    Returns DataFrame with columns:
    - driver: Driver number
    - driver_name: Driver full name
    - lap_number: Lap number in race
    - lap_time_seconds: Lap time in seconds
    - compound: Tire compound (SOFT, MEDIUM, HARD, INTERMEDIATE, WET)
    - tire_age: Number of laps on current tire set
    - pit_in_lap: Lap where pit stop occurred (NaN if not a pit lap)
    - is_green_flag: True if the entire lap was run under green flag racing
      (no safety car/VSC/yellow/red mixed in) -- FastF1's TrackStatus is a
      string of every status code that occurred during the lap (e.g. "124"
      means green, then yellow, then safety car all happened within that
      one lap), so "1" exactly is the only clean-racing case.
    - position: Race position after lap (from telemetry/race info)
    """
    laps = session.laps.copy()
    
    # Basic filtering: remove laps without valid times (accidents, DNF)
    laps = laps[laps["LapTime"].notna()].reset_index(drop=True)
    
    # Convert lap time to seconds
    laps["lap_time_seconds"] = laps["LapTime"].dt.total_seconds()
    
    # Extract driver info
    laps["driver"] = laps["Driver"]
    laps["driver_name"] = laps.apply(
        lambda row: session.get_driver(row["Driver"])["Abbreviation"]
        if pd.notna(row["Driver"])
        else "UNK",
        axis=1,
    )
    
    # Tire info
    laps["compound"] = laps["Compound"]
    laps["tire_age"] = laps["TyreLife"]
    
    # Pit detection: identify pit laps
    laps["pit_in_lap"] = (~laps["PitInTime"].isna()).astype(int)

    # Track status: "1" means green flag for the whole lap; any other code
    # mixed in (yellow/SC/VSC/red) means part of the lap wasn't clean racing
    laps["is_green_flag"] = laps["TrackStatus"] == "1"

    # Select final columns
    output = laps[
        [
            "driver",
            "driver_name",
            "LapNumber",
            "lap_time_seconds",
            "compound",
            "tire_age",
            "pit_in_lap",
            "is_green_flag",
        ]
    ].copy()

    output.columns = [
        "driver",
        "driver_name",
        "lap_number",
        "lap_time_seconds",
        "compound",
        "tire_age",
        "pit_in_lap",
        "is_green_flag",
    ]
    
    return output.reset_index(drop=True)


def extract_pit_stops(session: fastf1.core.Session) -> pd.DataFrame:
    """
    Extract real pit stop durations from a race session.

    FastF1 sets `PitInTime` on the lap a driver crosses the pit entry line
    and `PitOutTime` on the following lap, when they cross the pit exit
    line -- so a stop's duration is the gap between one lap's PitInTime and
    the *next* lap's PitOutTime for that driver. This is the standard
    proxy for total pit-lane time loss (used to calibrate
    race_simulator.py's DEFAULT_PIT_LOSS_SECONDS).

    Returns DataFrame with columns:
    - driver: Driver number
    - lap_number: The in-lap (lap on which PitInTime was recorded)
    - pit_duration_seconds: PitOutTime - PitInTime, in seconds
    """
    laps = session.laps.copy()
    rows = []

    for driver, driver_laps in laps.groupby("Driver"):
        driver_laps = driver_laps.sort_values("LapNumber").reset_index(drop=True)
        for i in range(len(driver_laps) - 1):
            pit_in = driver_laps.loc[i, "PitInTime"]
            pit_out = driver_laps.loc[i + 1, "PitOutTime"]
            if pd.notna(pit_in) and pd.notna(pit_out):
                rows.append(
                    {
                        "driver": driver,
                        "lap_number": driver_laps.loc[i, "LapNumber"],
                        "pit_duration_seconds": (pit_out - pit_in).total_seconds(),
                    }
                )

    return pd.DataFrame(rows, columns=["driver", "lap_number", "pit_duration_seconds"])


def get_race_data(year: int, gp: str) -> Tuple[pd.DataFrame, dict]:
    """
    Convenience function: load, extract, and return race data + metadata.
    
    Returns:
        (laps_df, race_info_dict)
    """
    session = load_race_session(year, gp, session_type="R")
    laps_df = extract_laps_df(session)
    
    race_info = {
        "year": year,
        "gp": gp,
        "event_name": session.event.get("EventName", "Unknown"),
        "circuit": session.event.get("CircuitShortName", "Unknown"),
        "num_laps": int(laps_df["lap_number"].max()),
        "num_drivers": laps_df["driver"].nunique(),
    }
    
    return laps_df, race_info
