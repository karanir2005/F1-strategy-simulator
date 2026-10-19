"""
Build data/tracks/track_params.json: per-circuit race parameters for the 2026
calendar (T-016, contract: team/contracts/track-optimizer.md section 1).

Run from the project root, fully offline:  python build_track_params.py

Sources (no FastF1 / network loads):
  - data/processed/   10 FastF1 2026 races (preferred where available)
  - data/staging/     41 OpenF1 races 2025+2026 (respect data_quality.usable)
  - data/jolpica/     cached 2026 schedule (calendar order, circuit ids) and
                      race results (lap-count cross-check)

Season rule: per circuit, 2026 data if a usable, non-wet 2026 race exists;
otherwise the most recent usable dry 2025 race. Lap count and pit loss are
track properties, so when the primary season has too few clean stops the
other season's stops are pooled in (said in `notes`).

Method per field (2026-10-19):
  total_laps        winner's lap count in Jolpica results of the newest
                    source race (cross-checked against race_info.total_laps;
                    the two agree for every circuit, see `notes` otherwise).
  pit_loss_s        median pit_lane_s of tyre-change stops that were NOT under
                    SC/VSC/red flag, 10 <= duration <= 120 s (the same >120 s /
                    red-flag exclusion as calibrate_pit_loss.py, which also
                    uses pit-lane duration, the quantity DEFAULT_PIT_LOSS_
                    SECONDS is calibrated on).
  pit_loss_noise_std_s  std of the same stops restricted to <= 60 s so that
                    one stuck car / penalty cannot dominate (the global
                    calibration std is 5.0 s with its looser 100 s cut).
  base_laptime_s    NOTE: follows race_simulator's convention, NOT "lap 1
                    fuel load": simulate_race adds fuel_effect() anchored to 0
                    at the FINAL lap and tyre delta relative to age 2, with
                    HARD as the zero pace offset. So base_laptime_s is the
                    HARD, age-2, final-lap-fuel (light-car) clean-air lap.
                    Computed from tire_model.clean_laps() laps (dry, green
                    flag, no in/out/restart laps) with road_gap_ahead_s >= 2 s
                    (clean air), each corrected by
                      t - predict_delta(compound, age) - pace_offset(compound)
                        - fuel_coef * (lap - total_laps)
                    using the fitted tire model (fuel_coef = fuel burn + mean
                    track evolution). Front-running = the 5 drivers with the
                    lowest median corrected pace (>= 10 clean laps each);
                    value = median of those drivers' corrected laps.
  max_stint_laps    longest observed stint (laps run between real tyre-change
                    pit stops, in-lap preceding out-lap; laps under SC count)
                    per dry compound, from dry races only. A compound with no
                    stint >= 5 laps in the primary season is filled from the
                    other season; stated in `notes`. Red-flag Monaco_2026 and
                    the Malaysia/Belgium_2025 phantom lap-2 tyre changes are
                    handled by splitting only at real in-lap -> out-lap pit pairs (plus compound
                    changes / tyre-age resets); a stop with no tyre-change record still splits.
  compounds         Pirelli C-number per HARD/MEDIUM/SOFT from laps.compound_c,
                    else race_info.compound_map; null where unknown.
"""

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))

from app.core import tire_model as tm  # noqa: E402  (read-only use)

OUT_PATH = ROOT / "data" / "tracks" / "track_params.json"
DIRS = {"processed": ROOT / "data" / "processed", "staging": ROOT / "data" / "staging"}

# circuit_id (Jolpica 2026 schedule) -> (display circuit, race-file prefix, extra note)
TRACKS = {
    "albert_park": ("Albert Park Circuit", "Australia"),
    "shanghai": ("Shanghai International Circuit", "China"),
    "suzuka": ("Suzuka Circuit", "Japan"),
    "miami": ("Miami International Autodrome", "Miami"),
    "villeneuve": ("Circuit Gilles Villeneuve", "Canada"),
    "monaco": ("Circuit de Monaco", "Monaco"),
    "catalunya": ("Circuit de Barcelona-Catalunya", "Barcelona"),
    "red_bull_ring": ("Red Bull Ring", "Austria"),
    "silverstone": ("Silverstone Circuit", "Britain"),
    "spa": ("Circuit de Spa-Francorchamps", "Belgium"),
    "hungaroring": ("Hungaroring", "Hungary"),
    "zandvoort": ("Circuit Zandvoort", "Netherlands"),
    "monza": ("Autodromo Nazionale Monza", "Italy"),
    "madring": ("Madring", "Madrid"),
    "baku": ("Baku City Circuit", "Azerbaijan"),
    "sepang": ("Sepang International Circuit", "Malaysia"),
    "marina_bay": ("Marina Bay Street Circuit", "Singapore"),
    "americas": ("Circuit of the Americas", "Austin"),
    "rodriguez": ("Autodromo Hermanos Rodriguez", "Mexico"),
    "interlagos": ("Autodromo Jose Carlos Pace (Interlagos)", "Brazil"),
    "vegas": ("Las Vegas Strip Circuit", "LasVegas"),
    "losail": ("Lusail International Circuit", "Qatar"),
    "yas_marina": ("Yas Marina Circuit", "AbuDhabi"),
}

# Races never used as a source regardless of the usable flag (T-009-qa).
EXCLUDE_RACES = {
    "Belgium_2025": "wet start, phantom lap-2 tyre changes",
    "Australia_2025": "wet race",
    "Britain_2025": "wet race",
}
WET_EXCLUDE_FRAC = 0.10  # share of laps on INTERMEDIATE/WET above which a race is not a dry-pace source
MIN_PIT_STOPS = 15  # clean stops needed from the primary season before pooling the other
DRY = ("SOFT", "MEDIUM", "HARD")

# Allocations the lap data does not carry (compound_c empty in most OpenF1 files).
# Only where a researched source exists: team/research/T-010-us-gp-2026.md
# (2025 Austin used C1/C3/C4, F1.com / Pirelli).
COMPOUND_OVERRIDES = {
    "Austin_2025": ({"HARD": "C1", "MEDIUM": "C3", "SOFT": "C4"}, "T-010 research note (2025 allocation)"),
}


def load_race(key: str):
    """(laps, pits, info, dir_name) for a race key like 'Japan_2026', or None if absent/unusable."""
    for dname in ("processed", "staging"):
        d = DIRS[dname]
        lp = d / f"{key}_laps.csv"
        if not lp.exists():
            continue
        info = json.loads((d / f"{key}_race_info.json").read_text(encoding="utf-8"))
        usable = (info.get("data_quality") or {}).get("usable", True)
        if not usable:
            continue
        laps = pd.read_csv(lp)
        laps["race_file"] = key
        pits = pd.read_csv(d / f"{key}_pit_stops.csv")
        return laps, pits, info, dname
    return None


def wet_fraction(laps: pd.DataFrame) -> float:
    return float(laps["compound"].isin(["INTERMEDIATE", "WET"]).mean())


def pit_durations(pits: pd.DataFrame) -> np.ndarray:
    ok = (
        pits["is_tyre_change"].astype(bool)
        & ~pits["pit_under_sc"].astype(bool)
        & ~pits["pit_under_vsc"].astype(bool)
        & ~pits["pit_under_red"].astype(bool)
        & pits["pit_lane_s"].between(10, 120)
    )
    return pits.loc[ok, "pit_lane_s"].to_numpy(dtype=float)


def stint_lengths(laps: pd.DataFrame) -> dict:
    """{compound: [stint laps]} split only at real in-lap -> out-lap tyre changes."""
    out = {c: [] for c in DRY}
    for _, g in laps.sort_values(["driver", "lap_number"]).groupby("driver"):
        g = g.reset_index(drop=True)
        prev_in = g["pit_in_lap"].shift(1).fillna(0).astype(int)
        # a real stop (in-lap -> tyre-change out-lap), or - because OpenF1 can miss
        # a pit pass - a dry compound change / tyre age falling back to a fresh set
        prev_pass = g["is_pit_pass_no_change"].astype(bool).shift(1, fill_value=False)
        # any stop whose tyre outcome is unrecorded (e.g. Singapore_2026 HAD/ALO) counts as a
        # change: a 60-lap stint across a recorded pit stop is a data gap, not tyre life
        real_stop = (prev_in == 1) & ~prev_pass
        comp_change = (g["compound"] != g["compound"].shift(1)) & g["compound"].shift(1).notna()
        age_reset = g["tire_age"] < g["tire_age"].shift(1) - 1
        new_stint = (real_stop | comp_change | age_reset).astype(int)
        sid = new_stint.cumsum()
        for _, s in g.groupby(sid):
            comp = s["compound"].mode()
            if len(comp) and comp.iloc[0] in DRY:
                out[comp.iloc[0]].append(int(len(s)))
    return out


def base_laptime(laps: pd.DataFrame, model: dict, total_laps: int):
    """(base_laptime_s, n_laps_used, n_drivers) - see module docstring."""
    clean = tm.clean_laps(laps)
    clean = clean[clean["road_gap_ahead_s"].fillna(0) >= 2.0]
    if clean.empty:
        return None, 0, 0
    fuel = model["_fuel_coef"]
    corr = []
    for comp, g in clean.groupby("compound"):
        delta = g["tire_age"].astype(int).map(lambda a: tm.predict_delta(model, comp, a))
        off = tm.compound_pace_offset(model, comp)
        corr.append(g["lap_time_seconds"] - delta - off - fuel * (g["lap_number"] - total_laps))
    clean = clean.assign(corr=pd.concat(corr))
    per = clean.groupby("driver")["corr"].agg(["median", "size"])
    per = per[per["size"] >= 10].sort_values("median")
    if per.empty:
        return None, 0, 0
    front = per.index[:5]
    sel = clean[clean["driver"].isin(front)]["corr"]
    return float(sel.median()), int(len(sel)), int(len(front))


def compound_map(laps: pd.DataFrame, info: dict) -> dict:
    m = {}
    for comp in DRY:
        vals = laps.loc[(laps["compound"] == comp) & laps["compound_c"].notna(), "compound_c"]
        if len(vals):
            m[comp] = vals.mode().iloc[0]
    for comp, c in (info.get("compound_map") or {}).items():
        m.setdefault(comp, c)
    return m


def main():
    model = tm.build_tire_model()
    sched = pd.read_csv(ROOT / "data" / "jolpica" / "derived" / "schedule.csv")
    sched = sched[sched["season"] == 2026].sort_values("round")
    results = pd.read_csv(ROOT / "data" / "jolpica" / "derived" / "results.csv")
    results = results[(results["session"] == "race") & (results["position"] == 1)]
    today = pd.Timestamp.today().normalize()

    rows = []
    for _, r in sched.iterrows():
        cid = r["circuit_id"]
        if cid not in TRACKS:
            raise SystemExit(f"calendar circuit {cid} missing from TRACKS mapping")
        circuit, prefix = TRACKS[cid]
        notes = []

        cands = {}
        for season in (2026, 2025):
            key = f"{prefix}_{season}"
            if key in EXCLUDE_RACES:
                notes.append(f"{key} excluded ({EXCLUDE_RACES[key]}).")
                continue
            loaded = load_race(key)
            if loaded is None:
                continue
            if wet_fraction(loaded[0]) > WET_EXCLUDE_FRAC:
                notes.append(f"{key} excluded (wet).")
                continue
            cands[season] = (key, *loaded)
        if not cands:
            raise SystemExit(f"no usable source for {cid}")

        primary = 2026 if 2026 in cands else 2025
        other = 2025 if primary == 2026 else None
        pkey, plaps, ppits, pinfo, pdir = cands[primary]
        sources = [pkey]
        if primary == 2025:
            notes.append(
                "Not raced (usable) in 2026: all data is from the 2025 race; base_laptime_s "
                "is 2025 pace (2025 cars) and may differ from 2026 pace."
            )

        # total laps: Jolpica winner laps for the primary race season/circuit
        w = results[(results["season"] == primary) & (results["circuit_id"] == cid)]
        total_laps = int(pinfo["total_laps"])
        if len(w):
            jl = int(w.iloc[0]["laps"])
            if jl != total_laps:
                notes.append(f"Jolpica winner laps {jl} vs race_info {total_laps}; used {jl}.")
            total_laps = jl
        # an upcoming-race circuit's 2025 lap count is a track property

        # pit loss
        d = list(pit_durations(ppits))
        if len(d) < MIN_PIT_STOPS and other is None and (2025 in cands and primary == 2026):
            other = 2025
        if len(d) < MIN_PIT_STOPS and 2025 in cands and primary == 2026:
            okey, _, opits, _, _ = cands[2025]
            d += list(pit_durations(opits))
            sources.append(okey)
            notes.append(
                f"Only {len(d) - len(pit_durations(opits))} clean green-flag stops in {pkey}; "
                f"pit loss pooled with {okey} (pit lane is a track property)."
            )
        d = np.array(d)
        if len(d) < 5:
            notes.append(f"Only {len(d)} clean pit stops: pit_loss is low confidence.")
        pit_loss = float(np.median(d))
        d60 = d[d <= 60]
        pit_std = float(d60.std(ddof=1)) if len(d60) > 2 else float("nan")

        # base lap time (primary season only)
        blap, nlaps, ndrv = base_laptime(plaps, model, total_laps)
        if blap is None:
            raise SystemExit(f"{cid}: no clean laps for base lap time")
        if nlaps < 80:
            notes.append(f"base_laptime_s from only {nlaps} clean-air laps: low confidence.")

        # stints
        stint_src = [(primary, cands[primary])]
        if 2025 in cands and primary == 2026:
            stint_src.append((2025, cands[2025]))
        if cid == "monaco":
            # red-flag Monaco_2026 stint tables are unreliable (T-009-qa)
            stint_src = [(2025, cands[2025])] if 2025 in cands else stint_src
            notes.append("max_stint_laps from Monaco_2025 (Monaco_2026 red-flag stints unreliable).")
        stints = {c: [] for c in DRY}
        filled = []
        for i, (season, (k, l, p, inf, dn)) in enumerate(stint_src):
            sl = stint_lengths(l)
            for c in DRY:
                got = [x for x in sl[c] if x >= 5]
                if got and (i == 0 or not stints[c]):
                    stints[c] = got
                    if i > 0:
                        filled.append(c)
                        sources.append(k) if k not in sources else None
                        notes.append(f"{c} stints not run in {pkey}; max_stint_laps filled from {k}.")
        max_stint = {c: int(max(v)) for c, v in stints.items() if v}
        if cid == "monaco" and "Monaco_2025" not in sources:
            sources.append("Monaco_2025")
        missing = [c for c in DRY if c not in max_stint]
        if missing:
            notes.append(f"No observed stints for {missing}; omitted from max_stint_laps.")

        # compounds
        cm = compound_map(plaps, pinfo)
        if not cm and 2025 in cands and primary == 2026:
            cm = compound_map(cands[2025][1], cands[2025][3])
        comp_out = {c: cm.get(c) for c in ("HARD", "MEDIUM", "SOFT")}
        if all(v is None for v in comp_out.values()) and pkey in COMPOUND_OVERRIDES:
            comp_out, why = COMPOUND_OVERRIDES[pkey]
            notes.append(f"compounds from {why}.")
        if all(v is None for v in comp_out.values()):
            comp_out = None
            notes.append("Pirelli C-numbers not in the data for this circuit (compounds=null).")

        rd = pd.Timestamp(r["date"])
        raced = bool(rd < today) and primary == 2026
        rows.append(
            {
                "track_id": f"{prefix.lower()}" if prefix != "LasVegas" else "las_vegas",
                "name": r["race_name"].replace("Grand Prix", "GP") + f" ({prefix})",
                "circuit": circuit,
                "total_laps": total_laps,
                "pit_loss_s": round(pit_loss, 2),
                "pit_loss_noise_std_s": round(pit_std, 2),
                "base_laptime_s": round(blap, 2),
                "compounds": comp_out,
                "max_stint_laps": max_stint,
                "source_season": primary,
                "source_races": sources,
                "on_2026_calendar": True,
                "already_raced_2026": raced,
                "notes": " ".join(notes)
                + f" base_laptime_s = HARD, age-2, final-lap-fuel clean-air pace of the 5 fastest "
                f"drivers ({nlaps} laps, race_simulator convention); pit stops n={len(d)}.",
                "_round": int(r["round"]),
            }
        )

    rows.sort(key=lambda x: x["_round"])
    for x in rows:
        x["notes"] = x["notes"].strip()
        del x["_round"]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{'track_id':14}{'laps':>5}{'pit':>7}{'std':>6}{'base':>8}  {'src':5} sources")
    for x in rows:
        print(
            f"{x['track_id']:14}{x['total_laps']:>5}{x['pit_loss_s']:>7.1f}{x['pit_loss_noise_std_s']:>6.1f}"
            f"{x['base_laptime_s']:>8.2f}  {x['source_season']}  {','.join(x['source_races'])}"
        )
    print(f"wrote {OUT_PATH} ({len(rows)} tracks)")


if __name__ == "__main__":
    main()
