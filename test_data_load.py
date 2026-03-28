"""
Quick test: Load 2026 China race, inspect, and save to CSV.
Run this from project root with: python test_data_load.py
"""

import sys
sys.path.insert(0, "backend")

import fastf1
from app.core.data_loader import get_race_data

# Enable cache
fastf1.Cache.enable_cache("cache")

# Load 2026 China race
print("Loading 2026 China Grand Prix...")
try:
    laps_df, race_info = get_race_data(2026, "China")
except Exception as e:
    print(f"Failed to load race data: {e}")
    raise

print(f"\n✓ {race_info['event_name']} ({race_info['year']})")
print(f"  Circuit: {race_info['circuit']}")
print(f"  Drivers: {race_info['num_drivers']}")
print(f"  Data shape: {laps_df.shape[0]} laps")

print("\nFirst 10 rows:")
print(laps_df.head(10))

print("\nTire compound distribution:")
print(laps_df["compound"].value_counts())

print("\nSaving to data/processed/China_2026_laps.csv...")
laps_df.to_csv("data/processed/China_2026_laps.csv", index=False)
print("✓ Saved!")
