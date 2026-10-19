"""
Calibrate DEFAULT_PIT_LOSS_SECONDS (race_simulator.py) and
DEFAULT_PIT_LOSS_NOISE_STD (monte_carlo.py) from real pit stop durations
across the pulled 2026 races.

Run this from project root with: python calibrate_pit_loss.py
"""

import sys

sys.path.insert(0, "backend")

import fastf1
import numpy as np
import pandas as pd

from app.core.data_loader import extract_pit_stops, load_race_session

fastf1.Cache.enable_cache("cache")

# Same race list as pull_races.py, so the pit-loss constant is calibrated
# from the same data the tire model was fitted on.
RACES = [
    (2026, "China"),
    (2026, "Australia"),
    (2026, "Japan"),
    (2026, "Monaco"),
    (2026, "Britain"),
    (2026, "Miami"),
    (2026, "Canada"),
    # (2026, "Spain") removed with pull_races.py's (T-002): it resolves to
    # the uncached Madrid race and would trigger a network pull. T-009.
    (2026, "Austria"),
    (2026, "Belgium"),
    (2026, "Hungary"),
]

all_durations = []

for year, gp in RACES:
    print(f"Loading {year} {gp} Grand Prix...")
    try:
        session = load_race_session(year, gp)
        pit_stops = extract_pit_stops(session)
    except Exception as e:
        print(f"  Failed to load {gp}: {e}")
        continue

    if pit_stops.empty:
        print("  No pit stops found")
        continue

    # Drop stops under 10s (formation-lap/red-flag artifacts, not real
    # pit-lane traversal) and over 100s (drive-through penalties, long
    # red-flag stops) -- neither represents a routine tire-change pit loss.
    clean = pit_stops[
        (pit_stops["pit_duration_seconds"] >= 10)
        & (pit_stops["pit_duration_seconds"] <= 100)
    ]
    print(f"  {len(pit_stops)} stops, {len(clean)} after filtering")
    all_durations.extend(clean["pit_duration_seconds"].tolist())

durations = np.array(all_durations)
print(f"\n{len(durations)} total clean pit stops across {len(RACES)} races")
print(f"  mean:   {durations.mean():.2f}s")
print(f"  median: {np.median(durations):.2f}s")
print(f"  std:    {durations.std():.2f}s")
print(f"  min:    {durations.min():.2f}s")
print(f"  max:    {durations.max():.2f}s")
print(f"  p10/p90: {np.percentile(durations, 10):.2f}s / {np.percentile(durations, 90):.2f}s")
