"""
Optimizer sanity check against real strategies, all 23 tracks (T-020).

Run from the repo root:  python check_optimizer_plans.py [--stops 3] [--n-sims 500]
Writes data/scorecard/optimizer_plans.json and prints the table.

For every circuit in data/tracks/track_params.json it runs
optimizer.optimize() with the shipped tyre model (build_tire_model()
defaults) and compares the top plan with what teams really ran in that
circuit's source race (the race the track parameters were built from:
2026 where it exists, else 2025):

  top plan     rank-1 plan, e.g. H34/S24 (compound initial + stint laps)
  winner       the race winner's real stints (tyre_stint split)
  top-10 stops stop counts of the top-10 finishers, e.g. 1x7 2x3
  gap          simulate_race(winner's plan) - optimizer top plan, same model
               and base lap: how much slower the model thinks the winner's
               real plan is. Small (< ~5 s) = the model sees it as
               near-optimal; large = the model disagrees with the winner.
  flag         races where the winner's stints are not comparable
               (red flag, wet start, fragmented) are marked and left out of
               the summary statistics.

NOT a validation of the optimizer against truth: real winners also reflect
safety cars, traffic, track position and undercuts that the single-car
engine does not model, and the winner's plan is only one of several good
ones. It is a sanity check for gross failures (e.g. soft-only plans, 5-lap
first stints), which is what T-019 found.
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
from app.core.optimizer import optimize  # noqa: E402
from app.core.race_simulator import Stint, simulate_race  # noqa: E402

TRACKS = ROOT / "data" / "tracks" / "track_params.json"
OUT = ROOT / "data" / "scorecard" / "optimizer_plans.json"
LETTER = {"SOFT": "S", "MEDIUM": "M", "HARD": "H", "INTERMEDIATE": "I", "WET": "W"}
FULL = {v: k for k, v in LETTER.items()}


def race_dir(key: str) -> Path:
    for d in ("processed", "staging"):
        if (ROOT / "data" / d / f"{key}_laps.csv").exists():
            return ROOT / "data" / d
    raise FileNotFoundError(key)


def driver_stints(laps: pd.DataFrame, driver: str) -> list:
    """[(compound, laps)] per real tyre stint (tyre_stint). NOT merged: two consecutive stints on the
    same compound are usually a real stop onto a fresh set (Austria H24/H28), not a data split."""
    d = laps[laps["driver"] == driver].sort_values("lap_number")
    out = []
    for _, g in d.groupby("tyre_stint", sort=True):
        comp = g["compound"].dropna()
        c = comp.iloc[0] if len(comp) else "UNKNOWN"
        n = int(g["lap_number"].max() - g["lap_number"].min() + 1)
        out.append((c, n))
    return out


def fmt(stints) -> str:
    return "/".join(f"{LETTER.get(c, '?')}{n}" for c, n in stints)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stops", type=int, default=3)
    ap.add_argument("--n-sims", type=int, default=500)
    ap.add_argument("--per-track", action="store_true", help="T-028: use per-track wear (writes optimizer_plans_per_track.json)")
    args = ap.parse_args()

    global_model = tm.build_tire_model()
    tracks = json.loads(TRACKS.read_text(encoding="utf-8"))
    rows = []
    for t in tracks:
        model = tm.model_for_track(global_model, t["track_id"] if args.per_track else None)
        key = t["source_races"][0]
        d = race_dir(key)
        laps = pd.read_csv(d / f"{key}_laps.csv")
        results = pd.read_csv(d / f"{key}_results.csv").sort_values("position")
        top10 = results.head(10)["driver"].tolist()
        winner = driver_stints(laps, top10[0])
        stops = [len(driver_stints(laps, drv)) - 1 for drv in top10]

        res = optimize(
            model, t["total_laps"], t["pit_loss_s"], t["base_laptime_s"], t["max_stint_laps"],
            typical_stint_laps=t.get("typical_stint_laps"), max_stops=args.stops, n_sims=args.n_sims, top_n=3,
        )
        best = res.plans[0]
        plan = [(s["compound"], s["laps"]) for s in best["stints"]]
        soft_share = sum(n for c, n in plan if c == "SOFT") / t["total_laps"]

        flags = []
        if laps["has_red"].astype(bool).any():
            flags.append("red flag")
        if laps["compound"].isin(["INTERMEDIATE", "WET"]).mean() > 0.10:
            flags.append("wet race")
        if any(c in ("INTERMEDIATE", "WET", "UNKNOWN") for c, _ in winner):
            flags.append("winner on wet/unknown tyre")
        if len(winner) > 4 or min(n for _, n in winner) < 6:
            flags.append("short/fragmented stints")
        gap = None
        if not any(c in ("INTERMEDIATE", "WET", "UNKNOWN") for c, _ in winner):
            wl = [Stint(c, n) for c, n in winner]
            run = sum(s.laps for s in wl)
            if run != t["total_laps"]:
                wl[-1] = Stint(wl[-1].compound, wl[-1].laps + t["total_laps"] - run)
            wt = simulate_race(wl, t["base_laptime_s"], model, t["pit_loss_s"]).total_time
            tt = simulate_race([Stint(c, n) for c, n in plan], t["base_laptime_s"], model, t["pit_loss_s"]).total_time
            gap = round(wt - tt, 1)
        rows.append(
            {
                "track_id": t["track_id"], "source": key, "total_laps": t["total_laps"],
                "top_plan": fmt(plan), "top_plan_stops": len(plan) - 1,
                "top_plan_compounds": sorted({LETTER[c] for c, _ in plan}),
                "soft_share": round(soft_share, 2),
                "runner_ups": [fmt([(s["compound"], s["laps"]) for s in p["stints"]]) for p in res.plans[1:3]],
                "winner": fmt(winner), "winner_stops": len(winner) - 1,
                "winner_compounds": sorted({LETTER.get(c, "?") for c, _ in winner}),
                "top10_stops": {int(k): int(v) for k, v in zip(*np.unique(stops, return_counts=True))},
                "modal_top10_stops": int(pd.Series(stops).mode().iloc[0]),
                "winner_plan_gap_s": gap, "flags": flags,
                "effective_caps": res.effective_caps,
            }
        )

    ok = [r for r in rows if not r["flags"]]
    summary = {
        "n_tracks": len(rows),
        "n_comparable": len(ok),
        "top_plan_stop_counts": {int(k): int(v) for k, v in zip(*np.unique([r["top_plan_stops"] for r in rows], return_counts=True))},
        "mean_soft_share_top_plan": round(float(np.mean([r["soft_share"] for r in rows])), 3),
        "n_top_plans_with_soft": int(sum("S" in r["top_plan_compounds"] for r in rows)),
        "n_top_plans_two_soft_stints": int(sum(r["top_plan"].count("S") >= 2 for r in rows)),
        "n_top_plans_min_stint_lt_10": int(sum(min(int(x[1:]) for x in r["top_plan"].split("/")) < 10 for r in rows)),
        "stops_match_modal_top10 (comparable)": int(sum(r["top_plan_stops"] == r["modal_top10_stops"] for r in ok)),
        "same_compound_set_as_winner (comparable)": int(sum(r["top_plan_compounds"] == r["winner_compounds"] for r in ok)),
        "median_winner_plan_gap_s (comparable)": float(np.median([r["winner_plan_gap_s"] for r in ok])),
        "max_winner_plan_gap_s (comparable)": float(np.max([r["winner_plan_gap_s"] for r in ok])),
        "winner_stops_hist (comparable)": {int(k): int(v) for k, v in zip(*np.unique([r["winner_stops"] for r in ok], return_counts=True))},
    }
    out_path = OUT.with_name("optimizer_plans_per_track.json") if args.per_track else OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"model": {"offsets": global_model["_compound_offsets"],
                                          "wear": {c: global_model[c]["age_coef"] for c in tm.DRY_COMPOUNDS},
                                          "fuel_coef": global_model["_fuel_coef"], "per_track": args.per_track},
                               "summary": summary, "tracks": rows}, indent=2), encoding="utf-8")

    print(f"{'track':<12} {'top plan':<16} {'S%':>3} | {'winner':<22} {'top10 stops':<12} {'gap s':>6}  flag")
    for r in rows:
        ts = " ".join(f"{k}x{v}" for k, v in r["top10_stops"].items())
        gap = "" if r["winner_plan_gap_s"] is None else f"{r['winner_plan_gap_s']:+.1f}"
        print(f"{r['track_id']:<12} {r['top_plan']:<16} {100 * r['soft_share']:>3.0f} | {r['winner']:<22} {ts:<12} {gap:>6}  {','.join(r['flags'])}")
    print()
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
