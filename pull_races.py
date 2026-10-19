"""
Pull multiple races into data/processed/ for tire model calibration.
Run this from project root with: python pull_races.py [--online] [--out DIR]

Writes, per race: {gp}_{year}_laps.csv (the only file the tire model
reads) plus results / weather / track_status / rcm / pit_stops sidecar
CSVs and a race_info.json (incl. the event's C-compound mapping). See
backend/app/core/data_loader.py for the column definitions.

OFFLINE BY DEFAULT (T-002, 2026-10-19): every race below is already in
cache/, so re-extracting should never touch the network. FastF1's offline
mode makes that a guarantee -- a race missing from the cache fails loudly
instead of silently starting a slow download. Pass --online to allow
network pulls (needed for any new race, e.g. T-009).
"""

import argparse
import sys

sys.path.insert(0, "backend")

import fastf1
from app.core.data_loader import get_race_tables, save_race_tables

# (year, gp) pairs — picked for variety in circuit character, since tire
# degradation behaves differently on street circuits vs high-speed tracks.
RACES = [
    (2026, "China"),
    (2026, "Australia"),  # semi-street, medium abrasion
    (2026, "Japan"),      # classic high-speed, historically hard on tires
    (2026, "Monaco"),     # low-speed street circuit, low degradation
    (2026, "Britain"),    # high-speed, high lateral load, high degradation
    (2026, "Miami"),
    (2026, "Canada"),
    (2026, "Austria"),
    (2026, "Belgium"),    # Spa
    (2026, "Hungary"),
]

# T-009 (2026-10-19): Spain/Barcelona/Madrid are NOT pulled here. FastF1 live
# timing is now paywalled (401), so every race outside the 10 cached above
# comes from OpenF1 + Jolpica via pull_openf1.py (-> data/staging/). Labels:
# the 2026 "Spanish Grand Prix" is Madrid (round 14, label "Madrid");
# Barcelona is the "Barcelona Grand Prix" (round 7, label "Barcelona").


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--online", action="store_true", help="allow network downloads")
    parser.add_argument("--out", default="data/processed", help="output directory")
    args = parser.parse_args()

    fastf1.Cache.enable_cache("cache")
    if not args.online:
        fastf1.Cache.offline_mode(True)

    for year, gp in RACES:
        print(f"\nLoading {year} {gp} Grand Prix...")
        try:
            tables = get_race_tables(year, gp)
        except Exception as e:
            print(f"  Failed to load {gp}: {e}")
            continue

        written = save_race_tables(tables, args.out, gp, year)
        laps = tables["laps"]
        info = tables["race_info"]
        print(
            f"  {info['event_name']}: {len(laps)} lap rows "
            f"({laps['lap_time_seconds'].isna().sum()} without LapTime), "
            f"{int(laps['is_tyre_change'].sum())} tyre changes, "
            f"{int(laps['is_pit_pass_no_change'].sum())} same-tyre pit passes, "
            f"compounds {info['compound_map']} -> {written['laps']}"
        )


if __name__ == "__main__":
    main()
