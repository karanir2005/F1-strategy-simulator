"""
Compound pace offsets + wear: how well does the data identify them? (T-020)

Run from the repo root:  python calibrate_tire_compounds.py [--reps 100]
Writes data/scorecard/tire_compound_fit.json and prints the tables below.

Why this exists: the tyre model's pace offsets used to be literature
placeholders because the fitted ones came out backwards (SOFT/MEDIUM
SLOWER than HARD). T-020 had to decide what to ship, and the decision
rests on three measurements this script reproduces:

1. Free vs ordered fit on the 2026 laps (16 races), 2025 laps (20 dry
   races) and both pooled (2025 at weight 0.5). "Ordered" = fit_joint_model
   (constrain=True): SOFT <= MEDIUM <= HARD pace, SOFT >= MEDIUM >= HARD >= 0
   wear. When the free fit already satisfies the ordering the two coincide.
2. Race-cluster bootstrap (resample whole races with replacement) of the free
   2026 fit: how many of the "inverted" orderings are statistically real, and
   how much of the ordered fit's boundary solution is noise.
3. The 2025 fit, which is the physically ordered one (SOFT faster and wearing
   faster) -- evidence that the 2026 inversion is a property of the 2026 data
   (energy-management driving, stint selection), not a pipeline bug.

The bootstrap resamples the cleaned laps once (no second-pass outlier
removal), so its point estimates differ from build_tire_model's by a few
thousandths; use it for spread, not for the shipped values.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))

from app.core import tire_model as tm  # noqa: E402

OUT = ROOT / "data" / "scorecard" / "tire_compound_fit.json"


def _row(fit: dict) -> dict:
    return {
        "off_SOFT": round(fit["offsets"]["SOFT"], 3),
        "off_MEDIUM": round(fit["offsets"]["MEDIUM"], 3),
        "wear_SOFT": round(fit["wear"]["SOFT"][0], 4),
        "wear_MEDIUM": round(fit["wear"]["MEDIUM"][0], 4),
        "wear_HARD": round(fit["wear"]["HARD"][0], 4),
        "fuel_coef": round(fit["fuel_coef"], 4),
    }


def fits_for(laps: pd.DataFrame, weights=None) -> dict:
    clean = tm.clean_laps(tm.assign_stint_ids(laps))
    w = None if weights is None else tm.lap_eras(clean).map(weights).astype(float)
    free = tm.fit_joint_model(clean, weights=w)
    con = tm.fit_joint_model(clean, constrain=True, weights=w)
    return {"n_clean_laps": int(len(clean)), "free": _row(free), "ordered": _row(con)}


def bootstrap(laps: pd.DataFrame, reps: int, seed: int = 20261019) -> dict:
    clean = tm.clean_laps(tm.assign_stint_ids(laps))
    files = sorted(clean["race_file"].unique())
    by_file = {f: clean[clean["race_file"] == f] for f in files}
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(reps):
        parts = []
        for j, f in enumerate(rng.choice(files, len(files))):
            part = by_file[f].copy()
            part["race_file"] = part["race_file"] + f"#{j}"
            parts.append(part)
        res = pd.concat(parts)
        free = tm.fit_joint_model(res)
        con = tm.fit_joint_model(res, constrain=True)
        rows.append(
            {
                "off_SOFT": free["offsets"]["SOFT"],
                "off_MEDIUM": free["offsets"]["MEDIUM"],
                "wear_SOFT": free["wear"]["SOFT"][0],
                "wear_MEDIUM": free["wear"]["MEDIUM"][0],
                "wear_HARD": free["wear"]["HARD"][0],
                "con_off_SOFT": con["offsets"]["SOFT"],
                "con_off_MEDIUM": con["offsets"]["MEDIUM"],
            }
        )
    b = pd.DataFrame(rows)
    summary = {c: {"mean": round(b[c].mean(), 4), "sd": round(b[c].std(), 4),
                   "p05": round(b[c].quantile(0.05), 4), "p95": round(b[c].quantile(0.95), 4)} for c in b}
    summary["P(wear_SOFT > wear_MEDIUM)"] = round(float((b.wear_SOFT > b.wear_MEDIUM).mean()), 3)
    summary["P(wear_MEDIUM > wear_HARD)"] = round(float((b.wear_MEDIUM > b.wear_HARD).mean()), 3)
    summary["P(SOFT faster than HARD, free)"] = round(float((b.off_SOFT < 0).mean()), 3)
    summary["P(MEDIUM faster than HARD, free)"] = round(float((b.off_MEDIUM < 0).mean()), 3)
    summary["P(SOFT faster than MEDIUM, free)"] = round(float((b.off_SOFT < b.off_MEDIUM).mean()), 3)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    args = ap.parse_args()

    l26 = tm.load_all_laps()
    l25 = tm.load_2025_staging_laps()
    both = pd.concat([l26, l25], ignore_index=True)
    out = {
        "n_races_2026": int(l26["race_file"].nunique()),
        "n_races_2025": int(l25["race_file"].nunique()),
        "fit_2026": fits_for(l26),
        "fit_2025": fits_for(l25),
        "fit_pooled_2025_w0.5": fits_for(both, weights={tm.ERA_2025: 0.5, tm.ERA_2026: 1.0}),
        "bootstrap_2026_free": bootstrap(l26, args.reps),
        "bootstrap_reps": args.reps,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
