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

T-002 (2026-10-19): laps CSVs now keep rows with no LapTime (SC laps), so
the lap filter is tire_model.clean_laps() -- the same filter the tire
model is fitted on (also drops lap 1, out-laps on new AND used sets, and
restart laps). With the old filter (green + not an in-lap) this script
gives std 2.76 s on the pre-T-002 data and 2.80 s on the new data; with
clean_laps it gives 1.03 s. Neither reproduces the 0.8 s currently in
monte_carlo.DEFAULT_LAP_TIME_NOISE_STD -- see team/handoffs/T-002-data-scientist.md.

T-028 (2026-10-23): also prints the MAD-based robust std (the 0.8 s constant was
described as MAD-robust, the script printed only the raw std, so they were never
comparable). Reads all of data/processed (16 races).
"""

import sys

sys.path.insert(0, "backend")

import glob

import numpy as np
import pandas as pd

from app.core.tire_model import (
    build_tire_model,
    clean_laps,
    compound_pace_offset,
    fuel_effect,
    predict_delta,
)

tire_model = build_tire_model()

residuals = []

for path in glob.glob("data/processed/*_laps.csv"):
    laps = pd.read_csv(path)
    clean = clean_laps(laps.assign(race_file=path))

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
# MAD-based robust std (1.4826 * median |x - median|): the same kind of estimate
# DEFAULT_LAP_TIME_NOISE_STD (monte_carlo.py) is documented as, so these two are
# directly comparable. The raw std is heavy-tail inflated (traffic, residual yellows).
mad_std = 1.4826 * np.median(np.abs(residuals - np.median(residuals)))
print(f"  raw std:        {residuals.std():.3f}s  (heavy-tail inflated, not comparable)")
print(f"  robust std MAD: {mad_std:.3f}s   <- compare with DEFAULT_LAP_TIME_NOISE_STD")
print(f"  p10/p90 abs: {np.percentile(np.abs(residuals), 10):.3f}s / {np.percentile(np.abs(residuals), 90):.3f}s")
