"""
Pull multiple races into data/processed/ for tire model calibration.
Run this from project root with: python pull_races.py
"""

import sys
sys.path.insert(0, "backend")

import fastf1
from app.core.data_loader import get_race_data

fastf1.Cache.enable_cache("cache")

# (year, gp) pairs — picked for variety in circuit character, since tire
# degradation behaves differently on street circuits vs high-speed tracks.
RACES = [
    (2026, "China"),      # already pulled, re-run is harmless (cache hits)
    (2026, "Australia"),  # semi-street, medium abrasion
    (2026, "Japan"),      # classic high-speed, historically hard on tires
    (2026, "Monaco"),     # low-speed street circuit, low degradation
    (2026, "Britain"),    # high-speed, high lateral load, high degradation
    (2026, "Miami"),
    (2026, "Canada"),
    (2026, "Spain"),      # Barcelona
    (2026, "Austria"),
    (2026, "Belgium"),    # Spa
    (2026, "Hungary"),
]

for year, gp in RACES:
    print(f"\nLoading {year} {gp} Grand Prix...")
    try:
        laps_df, race_info = get_race_data(year, gp)
    except Exception as e:
        print(f"  Failed to load {gp}: {e}")
        continue

    out_path = f"data/processed/{gp}_{year}_laps.csv"
    laps_df.to_csv(out_path, index=False)
    print(f"  {race_info['event_name']}: {laps_df.shape[0]} laps -> {out_path}")
