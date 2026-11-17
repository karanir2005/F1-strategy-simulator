"""
Leave-one-race-out validation of the full-grid simulation (T-034, rebuilt without the known leaks in T-060).

For each 2026 round R (2-20) everything the sim uses is rebuilt WITHOUT race R:
  - tyre model: refit without R's laps (as in T-034);
  - track entry (base lap, pit loss, stint caps): calibrate_grid_sim.rebuild_track (base lap from R's best Q1 time via a
    pole-to-race ratio of the other races, pit loss / stints from the same-track 2025 race or the other tracks' median).
    T-034 used track_params.json, which includes R (leak removed, size measured below);
  - interaction parameters: overtake logistic + lap-1 table refitted on the other races, R's track offset dropped
    (T-034 used race_interaction.json, which includes R);
  - calibration constants (pace scale/shrink/weights, pace noise, SC bunching, lap-1 spacing, strategy mix, held gap):
    fitted on the other races; the overtake intercept shift is chosen from pass-count / position-change targets that
    exclude R (calibrate_grid_sim.choose_config);
  - pace/DNF inputs from earlier rounds only, R's own Q1 times and grid (known before the start).
The sim ranks drivers by expected points; scored against the official finishing order (Spearman over all drivers,
top-3 hit rate, winner hit) next to the grid order and the Plackett-Luce predictor (walk-forward, grid mode).

Four configurations are scored, see calibrate_grid_sim.evaluate_modes: t034_leaky (as committed), t034_leak_fixed,
fitted, calibrated (+ calibrated_2d sensitivity). Standard errors are across races (n = 19).

    python validate_grid_sim.py [--n-sims 2000] [--rounds 2-20] [--table data/scorecard/grid_sim_selection_table.csv]

The pass-calibration table (sim passes / position change per race and (scale, shift)) is read from --table if it
exists, else recomputed (about 6 minutes; `python calibrate_grid_sim.py` writes it).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

import calibrate_grid_sim as C  # noqa: E402

OUT_PATH = ROOT / "data" / "scorecard" / "grid_sim_validation.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sims", type=int, default=2000)
    ap.add_argument("--rounds", default="2-20")
    ap.add_argument("--table", default=str(C.TABLE_PATH))
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--skip-2025", action="store_true", help="skip the 2025 out-of-sample season")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.rounds.split("-"))
    t0 = time.time()
    real = C.Real()
    st = C.Study(real, range(lo, hi + 1), verbose=False)
    tp = Path(a.table)
    if tp.exists():
        table = pd.read_csv(tp)
    else:
        table = C.selection_table(st)
        tp.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(tp, index=False)
    res = C.evaluate_modes(st, table, n_sims=a.n_sims)
    report = {"n_sims": a.n_sims, "rounds": [lo, hi], "modes": {}}
    for name, df in res["rows"].items():
        s = C.summarise_rows(df)
        report["modes"][name] = {"summary": s, "per_race": df.to_dict("records")}
        print(f"\n== {name}")
        for m in ("sim", "grid", "pl", "blend50", "blend25", "blend75"):
            if m in s:
                x = s[m]
                print(f"  {m:<5} Spearman {x['spearman']:.3f} +/- {x['spearman_se']:.3f}  top-3 {x['top3_hit_rate']:.3f} +/- {x['top3_se']:.3f}  "
                      f"winner {x['winner_hit_rate']:.3f} +/- {x['winner_se']:.3f}  n={x['n']}")
        for k in ("sim_minus_grid_spearman", "sim_minus_pl_spearman", "blend50_minus_pl_spearman", "blend50_minus_grid_spearman",
                  "passes", "position_change", "prob_scores"):
            if k in s:
                print(f"  {k}: {s[k]}")
    report["chosen_config_per_fold"] = res["chosen"]
    if not a.skip_2025:
        r25 = C.evaluate_2025(st, table, n_sims=a.n_sims)
        s25 = C.summarise_rows(r25["rows"])
        report["season_2025"] = {"summary": s25, "per_race": r25["rows"].to_dict("records")}
        print("\n== 2025 season (all-2026 constants, fully out of sample for the sim; PL config was tuned on 2025)")
        for m in ("sim", "grid", "pl"):
            if m in s25:
                x = s25[m]
                print(f"  {m:<5} Spearman {x['spearman']:.3f} +/- {x['spearman_se']:.3f}  top-3 {x['top3_hit_rate']:.3f} +/- {x['top3_se']:.3f}  "
                      f"winner {x['winner_hit_rate']:.3f} +/- {x['winner_se']:.3f}  n={x['n']}")
        for k in ("sim_minus_grid_spearman", "sim_minus_pl_spearman", "position_change"):
            if k in s25:
                print(f"  {k}: {s25[k]}")
        print("  sim passes/race:", float(r25["rows"]["sim_passes"].mean()))
    print(f"\n{time.time() - t0:.0f}s")
    if not a.no_write:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
