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
    ]
    
    return output.reset_index(drop=True)


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
