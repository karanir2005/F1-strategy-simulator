"""
Calibrate DEFAULT_LAP_TIME_NOISE_STD (monte_carlo.py) from real lap-time
residuals: for every clean (green-flag, non-pit) lap, predict lap time
with the same formula race_simulator.py uses (tire delta + fuel effect +
compound offset, relative to that driver's fastest lap of the race as
base_laptime), then look at how much actual laps deviate from that
prediction after removing each driver's average bias (their overall
race pace / track-position/traffic effects aren't modeled -- only the
lap-to-lap wobble around their own trend is what we want here).

Run this from project root with: python calibrate_lap_noise.py
"""

import sys

sys.path.insert(0, "backend")

import glob

import numpy as np
import pandas as pd

from app.core.tire_model import build_tire_model, compound_pace_offset, fuel_effect, predict_delta

tire_model = build_tire_model()

residuals = []

for path in glob.glob("data/processed/*_laps.csv"):
    laps = pd.read_csv(path)
    clean = laps[
        (laps["is_green_flag"])
        & (laps["pit_in_lap"] == 0)
        & (laps["compound"].isin(["SOFT", "MEDIUM", "HARD"]))
    ]

    for driver, driver_laps in clean.groupby("driver"):
        driver_laps = driver_laps.sort_values("lap_number")
        base_laptime = laps[laps["driver"] == driver]["lap_time_seconds"].min()
        total_laps = int(laps["lap_number"].max())

        driver_residuals = []
        for _, row in driver_laps.iterrows():
            tire_delta = predict_delta(tire_model, row["compound"], int(row["tire_age"]))
            fuel_delta = fuel_effect(tire_model, int(row["lap_number"]), total_laps)
            pace_offset = compound_pace_offset(tire_model, row["compound"])
            predicted = base_laptime + tire_delta + fuel_delta + pace_offset
            driver_residuals.append(row["lap_time_seconds"] - predicted)

        if len(driver_residuals) < 5:
            continue  # too few clean laps to trust this driver's bias estimate

        driver_residuals = np.array(driver_residuals)
        # remove this driver's average bias (base_laptime pick, overall race
        # pace) -- what's left is lap-to-lap wobble, which is what
        # DEFAULT_LAP_TIME_NOISE_STD is meant to model
        demeaned = driver_residuals - driver_residuals.mean()
        residuals.extend(demeaned.tolist())

residuals = np.array(residuals)
print(f"{len(residuals)} clean laps across {len(glob.glob('data/processed/*_laps.csv'))} races")
print(f"  std:    {residuals.std():.3f}s")
print(f"  p10/p90 abs: {np.percentile(np.abs(residuals), 10):.3f}s / {np.percentile(np.abs(residuals), 90):.3f}s")
