"""
Safety-car / VSC calibration (T-030) from the pulled per-lap data.

Pools data/processed (16 races of 2026) plus the *usable* races in data/staging
(2025 + 2026 OpenF1 pulls; processed wins when a race is in both). A lap is an
SC (VSC) lap when more than half of that lap's drivers carry has_sc (has_vsc).
Consecutive SC/VSC laps form one event.

Prints and writes data/tracks/safety_car_params.json, which
backend/app/core/safety_cars.py loads. Re-run if the race set changes.
Estimates:
  - p_any / n_events per race, event length (laps), start-lap distribution
  - lap-time multiple on SC/VSC laps vs the same lap's green median reference
  - pit loss ratio: (in-lap + out-lap excess vs field median) for stops made
    under SC/VSC divided by the same for green stops
"""
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "tracks" / "safety_car_params.json"


def load_races():
    races = {}
    for d in ("staging", "processed"):  # processed overwrites staging
        for f in sorted(glob.glob(str(ROOT / "data" / d / "*_laps.csv"))):
            key = os.path.basename(f)[:-9]
            info_f = f.replace("_laps.csv", "_race_info.json")
            info = json.load(open(info_f))
            dq = info.get("data_quality")
            if d == "staging" and dq is not None and not dq.get("usable", True):
                continue
            df = pd.read_csv(f)
            if "has_sc" not in df:
                continue
            races[key] = (df, int(info.get("total_laps") or df.lap_number.max()), d)
    return races


def events_from(mask):
    """mask: bool array indexed by lap-1 -> list of (start_lap, length)."""
    ev, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            ev.append((i + 1, j - i + 1))
            i = j + 1
        else:
            i += 1
    return ev


def main():
    races = load_races()
    n_races = len(races)
    sc_ev, vsc_ev = [], []  # (race, start, length, total)
    per_race = {}
    mults = {"sc": [], "vsc": []}
    excess = {"green": [], "sc": [], "vsc": []}
    for key, (df, total, _) in races.items():
        g = df.groupby("lap_number").agg(sc=("has_sc", "mean"), vsc=("has_vsc", "mean"), red=("has_red", "mean"))
        g = g.reindex(range(1, total + 1)).fillna(0)
        sc = (g.sc > 0.5).values
        vsc = (g.vsc > 0.5).values & ~sc
        red = (g.red > 0.5).values
        e_sc, e_vsc = events_from(sc), events_from(vsc)
        per_race[key] = {"total_laps": total, "sc": e_sc, "vsc": e_vsc, "red_laps": int(red.sum())}
        sc_ev += [(key, s, l, total) for s, l in e_sc]
        vsc_ev += [(key, s, l, total) for s, l in e_vsc]
        # lap-time multiple: SC lap median / race green median, with lap 1 and pit laps excluded
        d = df[(df.lap_time_seconds > 0) & (df.pit_in_lap == 0) & (df.pit_out_lap == 0) & (df.lap_number > 1)]
        green_laps = d[d.is_green_flag & ~d.lap_number.map(lambda n: bool(sc[n - 1] or vsc[n - 1] or red[n - 1]) if n <= total else True)]
        if len(green_laps) < 100:
            continue
        # per-driver green median removes pace differences; compare same driver
        gm = green_laps.groupby("driver").lap_time_seconds.median()
        for name, m in (("sc", sc), ("vsc", vsc)):
            laps = d[d.lap_number.map(lambda n: n <= total and bool(m[n - 1]) and not red[n - 1])]
            laps = laps[laps.lap_number.map(lambda n: not (n >= 2 and m[n - 2] is False)) | True]
            r = laps.lap_time_seconds / laps.driver.map(gm)
            # drop the first lap of an event (partly raced) and lap after red
            first = {s for s, _ in (e_sc if name == "sc" else e_vsc)}
            r = r[~laps.lap_number.isin(first)]
            mults[name] += list(r.dropna())
        # pit loss: in+out lap excess vs field median on the same lap numbers
        med = d.groupby("lap_number").lap_time_seconds.median()
        pit = df[(df.is_tyre_change) & (df.lap_time_seconds > 0)]
        for _, row in pit.iterrows():
            n = int(row.lap_number)
            if n >= total or n < 2:
                continue
            nxt = df[(df.driver == row.driver) & (df.lap_number == n + 1)]
            if nxt.empty or not (nxt.lap_time_seconds.iloc[0] > 0) or n + 1 not in med.index or n not in med.index:
                continue
            ex = (row.lap_time_seconds - med[n]) + (nxt.lap_time_seconds.iloc[0] - med[n + 1])
            if red[n - 1] or red[n]:
                continue
            regime = "sc" if (sc[n - 1] and sc[n]) else "vsc" if (vsc[n - 1] and vsc[n]) else ("skip" if (sc[n-1] or sc[n] or vsc[n-1] or vsc[n]) else "green")
            if regime == "skip":
                continue
            if regime == "green" and not (bool(row.is_green_flag) and bool(nxt.is_green_flag.iloc[0])):
                continue
            excess[regime].append(ex)

    out = {"n_races": n_races, "races": sorted(races)}
    for name, ev in (("sc", sc_ev), ("vsc", vsc_ev)):
        by_race = {}
        for r, s, l, t in ev:
            by_race.setdefault(r, []).append((s, l, t))
        lens = np.array([l for _, _, l, _ in ev])
        starts = np.array([s / t for _, s, _, t in ev])
        laps_total = sum(v[1] for v in races.values())
        # per-lap hazard of an event STARTING (excludes lap 1, where start-of-race SCs are lap-1 incidents)
        starts_abs = np.array([s for _, s, _, _ in ev])
        out[name] = {
            "p_any_per_race": len(by_race) / n_races,
            "events_per_race": len(ev) / n_races,
            "n_events": len(ev),
            "hazard_per_lap": len(ev) / laps_total,
            "length_laps_mean": float(lens.mean()),
            "length_laps_median": float(np.median(lens)),
            "length_laps_values": [int(x) for x in sorted(lens)],
            "start_frac_quartiles": [float(x) for x in np.percentile(starts, [25, 50, 75])],
            "starts_on_lap1_frac": float((starts_abs == 1).mean()),
            "lap_multiple_median": float(np.median(mults[name])) if mults[name] else None,
            "lap_multiple_n": len(mults[name]),
        }
        # early-race (first 10%) / mid / late hazard
        out[name]["start_hist_decile_counts"] = np.histogram(starts, bins=10, range=(0, 1))[0].tolist()
    gm = float(np.median(excess["green"]))
    out["pit_excess_s"] = {k: {"median": float(np.median(v)), "n": len(v)} for k, v in excess.items() if v}
    out["pit_loss_ratio"] = {
        k: float(np.median(excess[k]) / gm) for k in ("sc", "vsc") if excess[k]
    }
    out["per_race"] = per_race
    OUT.write_text(json.dumps(out, indent=1))
    for k in ("n_races", "sc", "vsc", "pit_excess_s", "pit_loss_ratio"):
        print(k, json.dumps(out[k], indent=1) if not isinstance(out[k], (int, float)) else out[k])


if __name__ == "__main__":
    main()
