"""
Calibrate DEFAULT_PIT_LOSS_SECONDS (race_simulator.py) and
DEFAULT_PIT_LOSS_NOISE_STD (monte_carlo.py) from real pit-lane durations.

Run this from project root with: python calibrate_pit_loss.py

T-028 (2026-10-23): reads every data/processed/*_pit_stops.csv (all 16 pulled
2026 races, FastF1 + OpenF1 sourced) instead of re-loading a hardwired list of
10 FastF1 sessions, which could not see the six OpenF1 races. Stop filter is
the same one build_track_params.pit_durations uses (real tyre change, not
under SC/VSC/red, 10-120 s), so this constant and the per-track pit_loss_s
come from the same stops. Monaco is reported separately and excluded from the
headline figures: its narrow pit lane produces a heavy queued-stop tail that
would skew a single global constant (see race_simulator.DEFAULT_PIT_LOSS_SECONDS).
The old 10-race run (all stops 10-100 s, incl. Monaco, FastF1 sessions) gave
369 stops / median 23.80 s / std 7.67 s.
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"


def clean_durations(pits: pd.DataFrame) -> np.ndarray:
    ok = (
        pits["is_tyre_change"].astype(bool)
        & ~pits["pit_under_sc"].astype(bool)
        & ~pits["pit_under_vsc"].astype(bool)
        & ~pits["pit_under_red"].astype(bool)
        & pits["pit_lane_s"].between(10, 120)
    )
    return pits.loc[ok, "pit_lane_s"].to_numpy(dtype=float)


def main() -> None:
    per_race, pooled = {}, []
    for path in sorted(glob.glob(str(PROCESSED / "*_pit_stops.csv"))):
        race = Path(path).name.replace("_pit_stops.csv", "")
        d = clean_durations(pd.read_csv(path))
        per_race[race] = d
        if not race.startswith("Monaco"):
            pooled.extend(d.tolist())
    for race, d in per_race.items():
        if len(d):
            print(f"  {race:<18} n={len(d):>3}  median {np.median(d):6.2f}  std {d.std():5.2f}")
    durations = np.array(pooled)
    print(f"\n{len(durations)} clean tyre-change stops across {len(per_race) - 1} races (Monaco excluded)")
    print(f"  mean:   {durations.mean():.2f}s")
    print(f"  median: {np.median(durations):.2f}s   <- DEFAULT_PIT_LOSS_SECONDS")
    print(f"  std:    {durations.std():.2f}s   <- DEFAULT_PIT_LOSS_NOISE_STD")
    print(f"  min/max: {durations.min():.2f}s / {durations.max():.2f}s")
    print(f"  p10/p90: {np.percentile(durations, 10):.2f}s / {np.percentile(durations, 90):.2f}s")


if __name__ == "__main__":
    main()
