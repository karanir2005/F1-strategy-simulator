"""
Tire degradation model: fits compound-specific lap-time-loss curves from
real race data, plus a "cliff" point where wear accelerates beyond the
smooth part of the curve.

RESOLVED CAVEAT (2026-08-14): earlier versions of this model showed
physically implausible degradation curves (all three compounds getting
*faster* with tire age for most of the stint) and, separately,
race_simulator.py validated 2.6-5.4% short of real race times because it
had no fuel-burn term. Root cause of both: FastF1's per-lap TrackStatus
wasn't being extracted or filtered, so safety car / VSC / yellow laps
(~13% of all laps -- artificially slow for reasons unrelated to tires or
fuel) were being fitted as if they were normal green-flag racing. Fixed
by adding is_green_flag to data_loader.py's output and filtering on it in
clean_laps() here. After the fix: fresh-tire-only fuel slope flipped from
the wrong sign to -0.054 s/lap (matches literature ballparks), degradation
curves became small and monotonic (physically sensible for durable
modern-era tires), and validate_replay.py error dropped to 0.25-2.90%
across 3 real races (was 2.6-5.4%). fit_shared_fuel_coef() still
underestimates relative to fit_fuel_coef_from_fresh_laps() -- see its
docstring -- but is no longer the blocking issue.

UPDATE (2026-08-14): user ran SOFT/MEDIUM/HARD single-stint, no-pit, 50-lap
comparisons in the frontend and the lines looked wrong -- nearly
identical. Root cause: predict_delta() is defined relative to each
compound's OWN age-2 baseline (always 0 there by construction), so the
model never captured that a fresh SOFT tyre is inherently faster than a
fresh HARD one -- that signal was destroyed by clean_laps()'s per-driver
demeaning, which mixes all compounds into one baseline. Added
fit_compound_pace_offsets() to estimate it properly, but it came back
backwards (SOFT slower than HARD) on a small, track-concentrated, likely
traffic-confounded sample -- see its docstring. build_tire_model() uses
COMPOUND_PACE_OFFSET_PLACEHOLDER (literature ballpark) instead, wired
into race_simulator.py's lap_time calculation. Revisit
fit_compound_pace_offsets once there's traffic/gap data to correct for.

UPDATE (2026-10-19, T-002): data + estimator rework. Three changes, each
measured with a leave-one-race-out (LORO) check -- fit on 9 races,
predict the within-stint lap-time shape (fuel + wear, stint-demeaned) of
every >=6-lap clean stint in the 10th:

1. Stints / cleaning. data_loader.py now emits real tyre stints
   (tyre_stint), out-lap flags and per-character track-status flags.
   clean_laps() drops out-laps on ANY set (the old tire_age == 1 rule let
   the out-laps of all 57 used-set stops -- which include part of the
   pit lane -- into the fit), lap 1, the first green lap after a
   SC/VSC/red (restart laps: +0.23 s median, +0.44 s mean vs. the
   driver's neighbouring green laps), NaN lap times, and laps with
   unknown tyre data. Local-yellow-only laps stay EXCLUDED: measured
   against the same neighbouring green laps they are +0.09 s median but
   +0.84 s mean / +1.56 s p90 (green: -0.00 / +0.10 / +0.82), i.e. mostly
   mild with a heavy incident tail, and they'd add only ~4% more laps.
   Cleaning alone did NOT fix the flat SOFT curve (old estimator on new
   data: SOFT still ~0 until age ~28).

2. Estimator (the actual N-3 root cause). The old fit regressed each
   compound separately on [1, lap_number, age, age^2], then THREW AWAY
   that compound's own lap_number (fuel) coefficient and simulated with
   a different, shared one from fit_fuel_coef_from_fresh_laps (-0.054
   vs. per-compound -0.026..-0.038). Fuel and wear are only separable
   *jointly*, so the wear curves were fitted against a fuel slope the
   simulator never uses. LORO: the old model under-predicted MEDIUM/HARD
   within-stint slopes by 0.03-0.04 s/lap (too little degradation) and
   SOFT's quadratic absorbed the fuel trend outright (age_coef -0.055 ==
   minus the fuel slope -> flat after the monotone clamp).
   fit_joint_model() replaces it with ONE regression, with driver-race
   fixed effects (within transformation), compound dummies, a
   lap_number slope PER RACE (fuel burn + that race's track evolution --
   Canada/Monaco evolve very differently from Austria/Hungary) and
   per-compound wear terms. The simulator's _fuel_coef is the lap-weighted
   mean of those per-race slopes, so wear and fuel are now fitted and
   simulated consistently.
   LORO within-stint RMSE, full pipeline (outlier removal, cliffs and
   placeholder extrapolation included): BEFORE (old code, old data)
   1.311 s -> AFTER (joint, linear) 1.157 s [joint quadratic: 1.163 s];
   slope bias (pred - actual) MEDIUM -0.024 -> -0.003, HARD -0.038 ->
   -0.003 s/lap. Estimator-only comparison on identical cleaned data, no
   cliffs: old 1.003 s -> joint quadratic 0.985 s -> joint linear 0.979 s.
   (Old code on the new CSVs gives exactly the BEFORE numbers: the old
   pipeline never used stint ids, so the stint fix alone can't move it.)
   Linear wear (age2_coef = 0) wins out of sample, so it's the default
   (build_tire_model(quadratic=True) restores the curve).

3. SOFT (T-004 QA N-3). The joint linear fit gives SOFT wear +0.0105
   s/lap (MEDIUM +0.038, HARD +0.050): it now increases with age, so a
   used SOFT set is slower than a new one. Be clear what that rests on:
   the joint QUADRATIC fit still has SOFT getting faster for ~13 laps and
   only rising after ~age 25 (U-shape), and LORO says held-out SOFT
   stints get faster than either model predicts (slope bias +0.11 s/lap,
   was +0.17). Per-race joint fits show why: SOFT wear is +0.04..+0.08
   s/lap at Australia, Austria, Belgium and Hungary, but ~0 / negative at
   Monaco (+0.015: short post-red-flag sprints in a train), Canada
   (-0.023 -- MEDIUM is negative there too, i.e. non-linear track
   evolution the per-race linear term can't absorb) and Miami (-0.012,
   49 laps), and Canada + Monaco hold ~half of all clean SOFT laps (SOFT
   is C5 at both). Fixing that properly needs per-track terms (ROADMAP
   Phase 3); a median-of-races pooling was tried and was worse out of
   sample (estimator-only LORO 0.989 s vs 0.979 s), so it's documented
   here rather than hacked around. A clean-air filter (road gap >= 1,
   1.5, 2 s) was also tried and barely moved any curve.

[Superseded by the 2026-10-20 UPDATE below -- offsets are now fitted, with an
ordering constraint.] Compound pace offsets stay on COMPOUND_PACE_OFFSET_PLACEHOLDER: the joint
fit still says SOFT (+0.38 s) and MEDIUM (+0.18 s) are SLOWER than HARD,
and keying on Pirelli C-number instead of the S/M/H label doesn't help
(C1 -0.47, C2 -0.28, C4 +0.18, C5 +0.25 s vs C3). Since that survives
the stint fix, out-lap/restart cleaning, per-race fuel and C-number
keying, it is not a data-cleaning artefact; candidates are traffic /
stint-situation selection and 2026 tyre/energy management. The fitted
values are exposed as model["_compound_offsets_fitted"] (diagnostic only).

UPDATE (2026-10-20, T-015 + T-020): more data, an era filter, and a tyre fit
that is physically ordered. Four changes, in the order they were decided.

1. Data (T-015). Six staged 2026 races -- Barcelona, Madrid, Italy,
   Azerbaijan, Singapore, Netherlands (OpenF1-sourced, usable=true) -- were
   promoted to data/processed/ (now 16 races, 14.2k clean laps vs 10 races /
   ~9.9k). Not promoted: Malaysia (wet start), Miami_2025 / Australia_2026
   staging (unusable, FastF1 Australia kept), the wet 2025 races. The
   promoted CSVs keep the staging columns season/era/source (a superset of
   the original schema; the original ten have no `era` column). Era filter:
   build_tire_model()/load_all_laps() keep ERA_2026 only; 2025 rows are
   dropped unless eras=(ERA_2025, ERA_2026) is passed, in which case the
   usable dry 2025 staging races are loaded too and era_weights can
   down-weight them. clean_laps additionally drops wet-track laps (Netherlands
   only, 129 laps) and stints that run THROUGH a red flag (Monaco 222 laps,
   Italy+Netherlands 36): tyre changes under a red flag are free and often
   unrecorded, and Monaco's cars show 55-67-lap "stints".

2. The free 2026 fit is not physical, and that is a property of the data,
   not of the pipeline. 16 races, free linear fit: offsets SOFT +0.13,
   MEDIUM +0.19 s vs HARD; wear SOFT 0.025 < MEDIUM 0.035 < HARD 0.036 s/lap.
   Race-cluster bootstrap (calibrate_tire_compounds.py, 100 reps): MEDIUM
   slower than HARD in 99% of resamples, SOFT wear above MEDIUM wear in 2%,
   SOFT faster than HARD in 16%. Things tried that do NOT change it: a clean-
   air filter (gap >= 1.5/2/3/5 s: SOFT +0.0..+0.19, MEDIUM +0.19..+0.30),
   exponential or quadratic per-race track evolution, a two-way fit with a
   fixed effect per (race, lap) (SOFT +0.10, MEDIUM +0.13), a flat-then-
   hinge wear shape, and a pit-stop "jump" estimator that ignores everything
   but the 2-3 laps either side of each stop (284 stops: a fresh SOFT after a
   MEDIUM stint gains ~0.5 s LESS than a fresh HARD). The same code on the
   20 usable dry 2025 races gives the textbook answer (SOFT -0.37, MEDIUM
   -0.05 vs HARD; wear 0.061 > 0.040 > 0.036), so the 2026 inversion is
   real in the 2026 data. Candidate causes, none proven: 2026 energy-
   management driving (pace set by the plan, not by the tyre), stint-
   selection effects, and the narrower 2026 tyres.

3. What ships (T-020). fit_joint_model(constrain=True): bounded least squares
   with SOFT <= MEDIUM <= HARD(=0) pace and SOFT >= MEDIUM >= HARD >= 0 wear.
   The data violate the ordering, so the solution sits on the boundary:
   MEDIUM offset 0.0 (equal to HARD), SOFT offset -0.12, wear 0.037 / 0.037 /
   0.035 s/lap, fuel -0.0507 s/lap. The constraint is statistically visible
   (RSS +0.44%, F = 30 on 2 df) but tiny in seconds (residual RMSE 0.896 ->
   0.898 s), and it is imposed against the data, not discovered in them.
   Why constrain rather than ship the free fit: the free fit makes MEDIUM
   strictly the worst tyre at every stint length; the constrained fit at
   least cannot rank a softer compound below a harder one on physics grounds.
   The optimizer check (check_optimizer_plans.py) is the evidence it is
   better: the model-time gap between each real top-10 finisher's strategy
   and the optimizer's best plan (smaller = the model agrees real teams are
   near-optimal) is median 15.8 s with the old placeholders and 2.3 s with
   this fit (17 comparable circuits; free fit 5.8 s). Scorecard (LORO, 16
   races): per-stint MAE 1.572 -> 1.554, relative MAE 0.765 -> 0.730,
   finishing-order rho 0.869 -> 0.870, strategy-pair accuracy 0.690 -> 0.661
   (n = 277, +-0.07: none of these differences is significant).
   2025 as a prior: pooling the 2025 races at weight 0.25/0.5/1 gives nearly
   the same ordered parameters (SOFT -0.11, MEDIUM 0.0, wear ~0.038-0.041),
   scores marginally better in LORO (rho 0.875-0.876, pair accuracy 0.682-
   0.686) and has the same optimizer regret, so it is NOT the default (era
   rule above) but is one argument away: build_tire_model(eras=(ERA_2025,
   ERA_2026), era_weights={ERA_2025: 0.5}).

4. COMPOUND_PACE_OFFSET_PLACEHOLDER is now only a fallback (fitted_offsets=
   False). COMPOUND_CLIFF_PENALTY_PLACEHOLDER is still a placeholder but is
   not binding: max_reliable_age is ~50 laps for every compound, past every
   stint length the optimizer's observed-maximum caps allow. A real
   per-compound cliff needs deep-stint data this set doesn't have.

UPDATE (2026-10-23, T-028): per-track wear with partial pooling. See
fit_track_wear() and model_for_track(). Each circuit's linear wear slope per
compound is fitted with the global offsets and per-race fuel slopes held
fixed, a ridge pull towards the global slope (prior sd tau = TRACK_WEAR_TAU
s/lap per lap of age, penalty sigma^2/tau^2, so a track with few stints stays
at the global value), and the same SOFT>=MEDIUM>=HARD>=0 ordering as the
global fit; a compound with < MIN_TRACK_STINTS stints at the track keeps the
global slope outright. Stored as model["_track_wear"][track_id]; the plain
model is unchanged and model_for_track(model, track_id) returns a copy whose
age_coef are the track's. Evaluation (validate_scorecard.py --per-track): plain
LORO is degenerate here (one 2026 race per circuit, so the held-out track has
no data and falls back to the global fit, identical by construction); the
test used instead is a split-half by driver inside the held-out race. See
team/handoffs/T-028-data-scientist.md for the numbers and for what is
shipped as default.

UPDATE (2026-10-25, T-043): pace offsets are now fitted WITHOUT the wear-ordering
coupling (build_tire_model offset_mode="pace_ordered", the default). Found by comparing
what the optimizer recommends with what top-10 teams actually ran in 34 dry races
(data/strategy_eval/, leave-one-race-out): in the 2026 fit the -0.123 s SOFT offset was an
artefact of the wear ordering. The FREE 2026 fit has SOFT +0.13 s / MEDIUM +0.19 s (slower
than HARD) and SOFT wear 0.025 < HARD 0.036; forcing wear SOFT >= MEDIUM >= HARD (T-020)
raises SOFT wear by ~0.012 s/lap and the joint solution compensates by moving the SOFT
offset by -0.25 s, onto the boundary at -0.123. Nothing in the lap times says SOFT is faster;
the optimizer nevertheless put SOFT in half of all 2026 plan laps (teams: 16%) and never used
MEDIUM. Now: offsets = the free fit's, projected onto SOFT <= MEDIUM <= HARD(=0) only
(project_pace_ordering), then wear refitted with those offsets fixed under the wear ordering.
2026: offsets (0, 0, 0), wear 0.0369 / 0.0369 / 0.0364, fuel -0.0513. 2025 data already
satisfy the ordering, so its offsets/wear are unchanged (SOFT -0.35, MEDIUM -0.12).
LORO scorecard (16 races): per-stint MAE 1.554 -> 1.547, rho 0.870 -> 0.870, pair accuracy
0.661 -> 0.675 (noise-level; no regression). Optimizer vs actual 2026 (14 dry races, clean
caps): plans containing the same compound set as the top-10 mode 0 -> 7, mode strategy in
top-3 plans 8 -> 11, SOFT share of plan laps 0.50 -> 0.17 (actual 0.16), MEDIUM 0 -> 0.32
(0.35). Tried and NOT shipped (same evaluation): cross-era per-track wear
(transfer_track_wear, no change in stop counts), a stint-life cliff (the lap-time residual
has no extra slope past the typical stint length: k = 0.00 in 2026, negative in 2025, so any k
would be tuned to team behaviour, not measured), SC-aware ranking, p60/p75 stint caps.

Known remaining gaps: MEDIUM is dominated by HARD in this model (same pace,
slightly more wear), so M appears in a top plan only when stint caps force
it; the ordered S/M wear is a tie, not a measurement; one global fuel slope
(per-race slopes range -0.003 .. -0.100, Madrid the outlier); no per-track
or C-number-specific compound behaviour (Phase 3).
"""

import glob
import json
import re
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

# resolved from this file's location, not the caller's cwd -- otherwise
# this breaks the moment something imports tire_model from a process
# launched with a different working directory (e.g. `uvicorn` started
# from backend/ instead of the repo root)
PROCESSED_DIR = str(Path(__file__).resolve().parents[3] / "data" / "processed")
STAGING_DIR = str(Path(__file__).resolve().parents[3] / "data" / "staging")

# Regulation eras (T-015). The 2026 cars/tyres/power units are a different
# machine from 2025's, and 2025 data is only ever an explicit, down-weighted
# prior -- never silently pooled into a 2026 fit. Rows carry an `era` column
# when they came through openf1_loader; the 10 original FastF1 CSVs don't, so
# the era is otherwise read from the `_<year>_` in the race file name.
ERA_2025 = "2025_regs"
ERA_2026 = "2026_regs"
DEFAULT_ERAS = (ERA_2026,)

# 2025 staging races that are never a dry-pace source (T-009-qa): wet races
# and OpenF1 data with missing laps. Mirrors build_track_params.EXCLUDE_RACES.
EXCLUDED_2025_RACES = ("Belgium_2025", "Australia_2025", "Britain_2025", "Miami_2025")

# SUPERSEDED (T-020): build_tire_model() now fits the offsets (ordered
# SOFT <= MEDIUM <= HARD) and uses them. Kept only as the fallback behind
# build_tire_model(fitted_offsets=False) so the old behaviour is reproducible.
# Seconds of laptime relative to HARD (negative = faster); rough F1
# ballpark, not from our data -- see the module docstring for why the free
# fit could not be used as-is.
COMPOUND_PACE_OFFSET_PLACEHOLDER = {"SOFT": -0.5, "MEDIUM": -0.2, "HARD": 0.0}

# PLACEHOLDER, not fitted from data (T-020: still true, and not binding --
# see the module docstring, point 4). A single global 0.3 s/lap cliff
# penalty (the old default) understates how differently compounds fall
# off: our own data barely has any laps run deep past a soft's natural
# life (see max_reliable_age's docstring -- SOFT samples drop to n=2 past
# age ~25), so there's nothing real to fit a per-compound severity from
# yet. These follow the general F1 ordering (softer = falls off harder
# once truly gone) as a ballpark, not a measurement.
COMPOUND_CLIFF_PENALTY_PLACEHOLDER = {"SOFT": 1.0, "MEDIUM": 0.5, "HARD": 0.2}


DRY_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")

# T-043: how build_tire_model() obtains the pace offsets, see its docstring and the module docstring's
# 2026-10-25 UPDATE. Switched by the strategy-vs-actual evaluation (data/strategy_eval/).
DEFAULT_OFFSET_MODE = "pace_ordered"


def era_of_year(year: int) -> str:
    return ERA_2026 if year >= 2026 else ERA_2025


def lap_eras(laps: pd.DataFrame) -> pd.Series:
    """Era of every lap row: the `era` column where filled, else from `_<year>_` in race_file."""
    from_name = laps["race_file"].astype(str).map(
        lambda p: era_of_year(int(m.group(1))) if (m := re.search(r"_(20\d\d)_", Path(p).name)) else ERA_2026
    )
    if "era" in laps.columns:
        return laps["era"].where(laps["era"].notna(), from_name)
    return from_name


def filter_eras(laps: pd.DataFrame, eras: Optional[Iterable[str]] = DEFAULT_ERAS) -> pd.DataFrame:
    """Keep only rows from the given regulation eras (None = keep everything)."""
    if eras is None:
        return laps
    return laps[lap_eras(laps).isin(set(eras)).values]


def load_all_laps(
    pattern: str = f"{PROCESSED_DIR}/*_laps.csv",
    eras: Optional[Iterable[str]] = DEFAULT_ERAS,
) -> pd.DataFrame:
    """
    Combine every pulled race into one DataFrame, tagged by source file.

    eras: regulation eras to keep (default 2026 only; None = everything the
        pattern matches). 2025 races only live in data/staging, so asking
        for ERA_2025 here ALSO pulls the usable dry 2025 staging races
        (load_2025_staging_laps) -- an explicit opt-in, see T-015.
    """
    frames = []
    for path in sorted(glob.glob(pattern)):
        df = pd.read_csv(path)
        df["race_file"] = path
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No lap files found matching {pattern}")
    laps = filter_eras(pd.concat(frames, ignore_index=True), eras)
    if eras is not None and ERA_2025 in set(eras):
        laps = pd.concat([laps, load_2025_staging_laps()], ignore_index=True)
    return laps


def load_2025_staging_laps(staging_dir: str = STAGING_DIR) -> pd.DataFrame:
    """Usable, dry 2025 staging races (OpenF1), tagged by race_file. Empty frame if none."""
    frames = []
    for path in sorted(glob.glob(f"{staging_dir}/*_2025_laps.csv")):
        key = Path(path).name.replace("_laps.csv", "")
        if key in EXCLUDED_2025_RACES:
            continue
        info_path = Path(path.replace("_laps.csv", "_race_info.json"))
        if info_path.exists():
            info = json.loads(info_path.read_text(encoding="utf-8"))
            if not (info.get("data_quality") or {}).get("usable", True):
                continue
        df = pd.read_csv(path)
        df["race_file"] = path
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _has_t002_schema(laps: pd.DataFrame) -> bool:
    """True for laps CSVs written by the T-002 data_loader (real tyre stints + flags)."""
    return {"tyre_stint", "pit_out_lap", "is_neutralised"}.issubset(laps.columns)


def assign_stint_ids(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Tag each lap with a per-driver stint number (stint_id) that only
    changes on a REAL tyre change.

    T-002 files carry data_loader's tyre_stint (FastF1 Stint split into
    real tyre changes vs same-tyre pit-lane passes), which is used as-is.
    Legacy pre-T-002 files only have tire_age, so they fall back to the
    old "new stint when tire_age == 1" rule -- which silently merges every
    stop onto a used set into the previous stint (57 of 343 real tyre
    changes in the 2026 data). Re-extract with pull_races.py to fix.
    """
    laps = laps.sort_values(["race_file", "driver", "lap_number"]).copy()
    if "tyre_stint" in laps.columns:
        laps["stint_id"] = laps["tyre_stint"].astype(int)
        return laps
    is_new_stint = laps["tire_age"] == 1
    laps["stint_id"] = is_new_stint.groupby(
        [laps["race_file"], laps["driver"]]
    ).cumsum()
    return laps


def clean_laps(laps: pd.DataFrame, drop_red_flag_stints: bool = True) -> pd.DataFrame:
    """
    Keep only laps that measure tyre + fuel pace (see the module docstring's
    2026-10-19 UPDATE for the measurements behind each rule):
      - lap time known, dry compound, tyre age known
      - not lap 1 (standing start)
      - not an in-lap (pit_in_lap) or out-lap (pit_out_lap -- on ANY set,
        new or used, and including same-tyre pit-lane passes)
      - whole lap green (is_green_flag, TrackStatus == "1"). Local-yellow
        laps stay excluded on purpose: heavy-tailed (+0.84 s mean).
      - not the first lap after a SC/VSC/red-flag lap (restart lap)
      - not a wet-track lap (rainfall flag; only Netherlands 2026 has any,
        129 otherwise-clean laps)
      - not part of a stint that runs THROUGH a red flag (drop_red_flag_stints,
        T-015): tyre changes under a red flag are free and often unrecorded
        or split into phantom stints, so a stint with a red-flag lap after
        its first lap has an unreliable tyre age (Monaco 2026: 222 clean
        laps, cars on "one set" for 55-67 laps; Italy/Netherlands: 36).
        A stint that STARTS on the red-flag out-lap (Italy: every car
        changed tyres at the lap-3 stoppage) is fine and is kept.
    Also removes each car's own race-average pace (laptime_demeaned) for
    the legacy per-compound fitters; fit_joint_model does its own exact
    within-driver-race transformation.

    Legacy (pre-T-002) files fall back to the old rule: tire_age > 1 as
    the out-lap proxy, which misses used-set out-laps.
    """
    laps = laps.sort_values(["race_file", "driver", "lap_number"])
    keep = (
        laps["lap_time_seconds"].notna()
        & laps["tire_age"].notna()
        & laps["compound"].isin(DRY_COMPOUNDS)
        & (laps["pit_in_lap"] == 0)
        & laps["is_green_flag"].astype(bool)
    )
    if _has_t002_schema(laps):
        prev_neutralised = (
            laps.groupby(["race_file", "driver"])["is_neutralised"]
            .shift(1)
            .astype("boolean")
            .fillna(False)
            .astype(bool)
        )
        keep &= (
            (laps["lap_number"] > 1)
            & (laps["pit_out_lap"] == 0)
            & ~prev_neutralised
        )
        if "tyre_data_missing" in laps.columns:
            keep &= ~laps["tyre_data_missing"].astype(bool)
        if drop_red_flag_stints and "has_red" in laps.columns:
            stint_key = [laps["race_file"], laps["driver"], laps["tyre_stint"]]
            first_lap = laps.groupby(stint_key, dropna=False)["lap_number"].transform("min")
            red_after_start = laps["has_red"].astype(bool) & (laps["lap_number"] > first_lap)
            keep &= ~red_after_start.groupby(stint_key, dropna=False).transform("any").astype(bool)
    else:
        keep &= laps["tire_age"] > 1
    if "rainfall" in laps.columns:
        keep &= ~laps["rainfall"].fillna(False).astype(bool)
    clean = laps[keep].copy()
    clean["laptime_demeaned"] = clean["lap_time_seconds"] - clean.groupby(
        ["race_file", "driver"]
    )["lap_time_seconds"].transform("mean")
    return clean


def _joint_design(clean: pd.DataFrame, quadratic: bool, with_offsets: bool = True) -> Tuple[pd.DataFrame, list]:
    """Design matrix for fit_joint_model (before the within transformation)."""
    X = pd.DataFrame(index=clean.index)
    for compound in (("SOFT", "MEDIUM") if with_offsets else ()):  # HARD is the reference level
        X[f"off_{compound}"] = (clean["compound"] == compound).astype(float)
    for race in sorted(clean["race_file"].unique()):
        X[f"fuel_{race}"] = (clean["race_file"] == race) * clean["lap_number"].astype(float)
    for compound in DRY_COMPOUNDS:
        is_c = (clean["compound"] == compound).astype(float)
        X[f"age_{compound}"] = is_c * clean["tire_age"]
        if quadratic:
            X[f"age2_{compound}"] = is_c * clean["tire_age"] ** 2
    X = X.loc[:, X.abs().sum() > 0]  # drop compounds/races absent from this sample
    return X, list(X.columns)


def _ordering_transform(cols: list) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reparametrisation T (n_cols x n_theta) and lower bounds so that
    beta = T @ theta enforces the physical ordering with theta >= 0:
        offsets   SOFT <= MEDIUM <= HARD(=0):  off_MEDIUM = -p,  off_SOFT = -p - q
        wear      SOFT >= MEDIUM >= HARD >= 0: age_HARD = a, age_MEDIUM = a + b,
                                               age_SOFT = a + b + c
    Every other column (per-race fuel slopes) passes through unbounded.
    Linear wear only (age2 columns are not supported here).
    """
    idx = {c: i for i, c in enumerate(cols)}
    n = len(cols)
    passthrough = [c for c in cols if c.startswith("fuel_")]
    names = passthrough + ["p", "q", "a", "b", "c"]
    T = np.zeros((n, len(names)))
    lower = np.full(len(names), -np.inf)
    col_of = {nm: j for j, nm in enumerate(names)}
    for c in passthrough:
        T[idx[c], col_of[c]] = 1.0
    for nm in ("p", "q", "a", "b", "c"):
        lower[col_of[nm]] = 0.0
    if "off_MEDIUM" in idx:
        T[idx["off_MEDIUM"], col_of["p"]] = -1.0
    if "off_SOFT" in idx:
        T[idx["off_SOFT"], col_of["p"]] = -1.0
        T[idx["off_SOFT"], col_of["q"]] = -1.0
    if "age_HARD" in idx:
        T[idx["age_HARD"], col_of["a"]] = 1.0
    if "age_MEDIUM" in idx:
        T[idx["age_MEDIUM"], col_of["a"]] = 1.0
        T[idx["age_MEDIUM"], col_of["b"]] = 1.0
    if "age_SOFT" in idx:
        T[idx["age_SOFT"], col_of["a"]] = 1.0
        T[idx["age_SOFT"], col_of["b"]] = 1.0
        T[idx["age_SOFT"], col_of["c"]] = 1.0
    return T, lower


def fit_joint_model(
    clean: pd.DataFrame,
    quadratic: bool = False,
    constrain: bool = False,
    weights: Optional[pd.Series] = None,
    fixed_offsets: Optional[dict] = None,
) -> dict:
    """
    One least-squares fit of lap time on
        driver-race fixed effect                       (car/driver/track pace)
      + compound offset (SOFT, MEDIUM vs HARD)         (fresh-tyre pace)
      + per-race slope * lap_number                    (fuel burn + track evolution)
      + per-compound wear: age_coef * age [+ age2_coef * age^2]
    The fixed effects are removed exactly by demeaning y AND every
    regressor within (race_file, driver) -- unlike the legacy fitters,
    which demeaned y only. Wear is identified from tyre age resetting at
    each stop while lap_number doesn't; see the module docstring for why
    this replaced the per-compound fits (T-002, N-3).

    constrain (T-020): bounded least squares with the physical ordering
    SOFT <= MEDIUM <= HARD(=0) on the pace offsets and SOFT >= MEDIUM >=
    HARD >= 0 on the (linear) wear slopes -- see _ordering_transform. If the
    data already satisfy the ordering the constrained and free fits are
    identical; where it doesn't the solution sits on the boundary (e.g.
    two compounds get the same slope). Linear wear only.

    fixed_offsets (T-043): {"SOFT": s, "MEDIUM": m} pace offsets taken as known
    (subtracted from lap time, no offset columns fitted); only wear and fuel are
    estimated. Used by build_tire_model(offset_mode="pace_ordered").

    weights: optional per-lap weight (aligned to clean.index), e.g. to
    down-weight a prior era. Must be constant within a (race_file, driver)
    group so the exact within transformation stays valid (era weights are).

    Returns {"coef": {name: value}, "fuel_coef": lap-weighted mean of the
    per-race lap_number slopes, "fuel_by_race": {...}, "offsets": {...},
    "wear": {compound: (age_coef, age2_coef)}, "residual": Series of the
    full-model residual per lap, "partial_wear_residual": Series of
    y - FE - offsets - fuel (i.e. what's left for the wear curve),
    "constrained": bool}.
    """
    if constrain and quadratic:
        raise ValueError("ordering constraints are only implemented for linear wear")
    X, cols = _joint_design(clean, quadratic, with_offsets=fixed_offsets is None)
    groups = [clean["race_file"], clean["driver"]]
    y = clean["lap_time_seconds"].astype(float)
    if fixed_offsets is not None:
        y = y - clean["compound"].map(lambda c: float(fixed_offsets.get(c, 0.0))).astype(float)
    Xw = X - X.groupby(groups).transform("mean")
    yw = y - y.groupby(groups).transform("mean")
    sw = np.ones(len(clean)) if weights is None else np.sqrt(np.asarray(weights, dtype=float))
    if constrain:
        T, lower = _ordering_transform(cols)
        theta = lsq_linear(
            (Xw.values @ T) * sw[:, None], yw.values * sw,
            bounds=(lower, np.full(len(lower), np.inf)), method="bvls",
        ).x
        beta = T @ theta
    else:
        beta, *_ = np.linalg.lstsq(Xw.values * sw[:, None], yw.values * sw, rcond=None)
    coef = dict(zip(cols, beta))

    fitted = X.values @ beta
    fe = (y - fitted).groupby(groups).transform("mean")
    residual = y - fitted - fe
    wear_cols = [c for c in cols if c.startswith("age")]
    wear_part = X[wear_cols].values @ np.array([coef[c] for c in wear_cols])
    partial_wear_residual = residual + wear_part

    fuel_by_race = {c[len("fuel_"):]: coef[c] for c in cols if c.startswith("fuel_")}
    race_laps = clean["race_file"].value_counts()
    race_w = pd.Series(sw ** 2, index=clean.index).groupby(clean["race_file"]).mean()
    fuel_coef = float(
        np.average(
            list(fuel_by_race.values()),
            weights=[race_laps[r] * race_w[r] for r in fuel_by_race],
        )
    )
    offsets = {"HARD": 0.0}
    for compound in ("SOFT", "MEDIUM"):
        offsets[compound] = (
            float(fixed_offsets.get(compound, 0.0)) if fixed_offsets is not None else float(coef.get(f"off_{compound}", 0.0))
        )
    wear = {
        compound: (float(coef.get(f"age_{compound}", 0.0)), float(coef.get(f"age2_{compound}", 0.0)))
        for compound in DRY_COMPOUNDS
    }
    return {
        "coef": coef,
        "fuel_coef": fuel_coef,
        "fuel_by_race": fuel_by_race,
        "offsets": offsets,
        "wear": wear,
        "residual": residual,
        "partial_wear_residual": partial_wear_residual,
        "constrained": bool(constrain),
    }


def fit_fuel_and_tire_model(clean: pd.DataFrame, compound: str) -> dict:
    """
    Jointly regress demeaned lap time on lap_number (a proxy for fuel
    burned off, since fuel drops continuously across the race) and on
    tire_age + tire_age^2 (the wear curve). Fitting both at once, instead
    of detrending fuel first and fitting tire wear second, is what lets
    the model tell them apart: tire_age resets every stint while
    lap_number never does, so within a single regression the two
    variables aren't collinear enough to fool each other.
    """
    sub = clean[clean["compound"] == compound]
    X = np.column_stack(
        [
            np.ones(len(sub)),
            sub["lap_number"],
            sub["tire_age"],
            sub["tire_age"] ** 2,
        ]
    )
    y = sub["laptime_demeaned"].values
    (intercept, fuel_coef, age_coef, age2_coef), *_ = np.linalg.lstsq(X, y, rcond=None)
    return {
        "intercept": intercept,
        "fuel_coef": fuel_coef,
        "age_coef": age_coef,
        "age2_coef": age2_coef,
    }


def tire_only_residual(clean: pd.DataFrame, compound: str, fit: dict) -> pd.Series:
    """Lap time with the fuel/pace component removed, leaving only tire-age-driven variation."""
    sub = clean[clean["compound"] == compound]
    fuel_component = fit["intercept"] + fit["fuel_coef"] * sub["lap_number"]
    return sub["laptime_demeaned"] - fuel_component


def remove_outliers(
    clean: pd.DataFrame, residual: pd.Series, z_thresh: float = 2.5
) -> pd.DataFrame:
    """Drop laps whose fuel-adjusted residual is a z-score outlier within its tire_age bucket."""
    df = clean.copy()
    df["residual"] = residual
    df = df.dropna(subset=["residual"])

    grouped = df.groupby("tire_age")["residual"]
    mean = grouped.transform("mean")
    std = grouped.transform("std")
    z = (df["residual"] - mean).abs() / std

    # a bucket with std == 0 or NaN (e.g. a single-lap bucket) can't be
    # z-scored -- keep everything in it rather than filtering by an
    # undefined/divide-by-zero z
    keep = (std == 0) | std.isna() | (z < z_thresh)
    return df[keep]


def detect_cliff(
    filtered: pd.DataFrame, fit: dict, threshold: float = 0.3, sustained_for: int = 3
) -> Optional[int]:
    """
    Find the tire_age where the real (binned) average residual first
    exceeds the fitted quadratic by more than `threshold` seconds AND
    stays above it for `sustained_for` consecutive ages -- the point
    where wear stops being "smooth" and starts falling off a cliff.
    Requiring consecutive bins avoids a single noisy age (small sample,
    one bad lap) from being mistaken for a real cliff.
    """
    binned = filtered.groupby("tire_age")["residual"].mean()
    predicted = fit["age_coef"] * binned.index + fit["age2_coef"] * binned.index**2
    above = (binned.values - predicted) > threshold

    for i in range(len(above) - sustained_for + 1):
        if above[i : i + sustained_for].all():
            return int(binned.index[i])
    return None


def fit_fuel_coef_from_fresh_laps(clean: pd.DataFrame, max_age: int = 3) -> float:
    """
    Estimate the fuel/race-progress coefficient using only fresh-tire laps
    (tire_age <= max_age), where degradation is close enough to zero for
    every compound that it can't meaningfully confound the fit. Simpler
    and more direct than fit_shared_fuel_coef's joint regression, and
    avoids that function's failure mode where flexible per-compound
    quadratics can soak up some of the shared fuel trend for themselves.
    """
    fresh = clean[clean["tire_age"] <= max_age]
    slope = np.polyfit(fresh["lap_number"], fresh["laptime_demeaned"], 1)[0]
    return float(slope)


def fit_shared_fuel_coef(clean: pd.DataFrame) -> float:
    """
    Estimate one lap_number (fuel) coefficient shared across all
    compounds, fit jointly with separate per-compound tire_age
    quadratics (intercept + age + age^2 per compound). Fuel burn doesn't
    depend on which tire is fitted, so estimating it once here -- instead
    of taking three separate, noisier per-compound estimates -- is both
    more principled and lower-variance.

    SUPERSEDED (2026-08-14): before data_loader.py extracted track status,
    this returned ~-0.01 s/lap (too small) and a fresh-tire-only check
    gave the wrong sign entirely -- safety car / VSC laps were being
    fitted as if they were normal green-flag laps. Now that clean_laps()
    filters to is_green_flag laps, this still comes out smaller (~-0.033)
    than fit_fuel_coef_from_fresh_laps (~-0.054), which build_tire_model()
    uses instead -- likely because the per-compound quadratics here still
    have enough freedom to absorb part of the shared trend. Kept around
    as a cross-check / for future refinement.
    """
    compounds = sorted(clean["compound"].dropna().unique())
    compounds = [c for c in compounds if c in ("SOFT", "MEDIUM", "HARD")]
    k = len(compounds)
    sub = clean[clean["compound"].isin(compounds)]
    n = len(sub)

    X = np.zeros((n, 3 * k + 1))
    tire_age = sub["tire_age"].values
    for i, compound in enumerate(compounds):
        mask = (sub["compound"] == compound).values
        X[mask, i] = 1.0
        X[mask, k + i] = tire_age[mask]
        X[mask, 2 * k + i] = tire_age[mask] ** 2
    X[:, 3 * k] = sub["lap_number"].values

    y = sub["laptime_demeaned"].values
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(coeffs[3 * k])


def fit_compound_pace_offsets(clean: pd.DataFrame, model: dict) -> dict:
    """
    Estimate each compound's inherent pace offset relative to HARD (e.g.
    "SOFT is ~0.5s/lap faster than HARD when fresh"), by pairing up laps
    from the SAME driver in the SAME race: subtract out the fitted fuel
    and tire-degradation components from every lap, average what's left
    per (race, driver, compound), then diff SOFT/MEDIUM against HARD for
    every driver who ran both in that race. Pairing within the same
    driver+race cancels out car and track pace, in principle isolating
    the compound effect.

    NOT CURRENTLY TRUSTED (2026-08-14): this gives SOFT as ~0.59s/lap
    SLOWER than HARD -- backwards from well-established tire physics (a
    fresh soft tyre is reliably faster than a fresh hard one). The paired
    sample is small (37 driver-races for SOFT vs HARD) and dominated by a
    couple of tracks (Monaco, Canada), and there's a confound this
    function can't correct for: we have no data on traffic/dirty air, and
    SOFT-tyre stints are disproportionately run in different race
    situations than HARD-tyre stints. build_tire_model() uses
    COMPOUND_PACE_OFFSET_PLACEHOLDER instead until that's fixed -- likely
    needs per-track offsets and/or traffic-corrected lap data, not just
    more races of the same kind.
    """
    compounds = ("SOFT", "MEDIUM", "HARD")
    sub = clean[clean["compound"].isin(compounds)].copy()
    fuel_coef = model["_fuel_coef"]

    def adjusted_pace(row) -> float:
        return (
            row["lap_time_seconds"]
            - fuel_coef * row["lap_number"]
            - predict_delta(model, row["compound"], int(row["tire_age"]))
        )

    sub["adjusted_pace"] = sub.apply(adjusted_pace, axis=1)
    per_driver_compound = sub.groupby(["race_file", "driver", "compound"])[
        "adjusted_pace"
    ].mean().unstack("compound")

    offsets = {"HARD": 0.0}
    for compound in ("SOFT", "MEDIUM"):
        if compound not in per_driver_compound.columns or "HARD" not in per_driver_compound.columns:
            offsets[compound] = 0.0
            continue
        paired = (per_driver_compound[compound] - per_driver_compound["HARD"]).dropna()
        offsets[compound] = float(paired.mean()) if len(paired) else 0.0
    return offsets


def compound_pace_offset(model: dict, compound: str) -> float:
    """Seconds of inherent pace offset for this compound relative to HARD (negative = faster)."""
    return model["_compound_offsets"][compound]


def _drop_residual_outliers(
    clean: pd.DataFrame, residual: pd.Series, z_thresh: float = 2.5
) -> pd.DataFrame:
    """
    Drop laps whose joint-model residual is a z-score outlier within its
    (compound, tire_age) bucket -- same rule as remove_outliers(), applied
    to the joint fit. Buckets with an undefined std keep every lap.
    """
    df = clean.assign(_residual=residual)
    grouped = df.groupby(["compound", "tire_age"])["_residual"]
    mean = grouped.transform("mean")
    std = grouped.transform("std")
    z = (df["_residual"] - mean).abs() / std
    keep = (std == 0) | std.isna() | (z < z_thresh)
    return df[keep].drop(columns="_residual")


# --- per-track wear (T-028) ------------------------------------------------
# Prior sd of a track's wear slope around the global slope, s/lap per lap of
# age. Chosen by the split-half-by-driver scorecard (validate_scorecard.py
# --per-track), not by eye -- see the T-028 handoff.
TRACK_WEAR_TAU = 0.01
MIN_TRACK_STINTS = 3  # fewer stints of a compound at a track -> keep the global slope

_TRACK_ALIASES = {"lasvegas": "las_vegas"}


def track_id_of_race(race_file: str) -> str:
    """track_params.json track_id for a laps file: 'AbuDhabi_2026_laps.csv' -> 'abudhabi'."""
    name = re.split(r"_20\d\d", Path(str(race_file)).name)[0].lower()
    return _TRACK_ALIASES.get(name, name)


def fit_track_wear(
    filtered: pd.DataFrame,
    final: dict,
    tau: float = TRACK_WEAR_TAU,
    min_stints: int = MIN_TRACK_STINTS,
) -> dict:
    """
    Per-circuit linear wear slopes, partially pooled towards the global fit.

    For each track (all laps whose race_file maps to the same track_id), take
    the global fit's offsets and per-race fuel slopes as given, remove them
    from lap time, remove the driver-race fixed effects by the same within
    transformation as fit_joint_model, and fit  y = sum_c a_c * age_c  by
    bounded least squares with  a = (A, A+B, A+B+C), A,B,C >= 0  (the
    SOFT >= MEDIUM >= HARD >= 0 ordering) plus a ridge row per compound
    pulling a_c towards the global slope g_c with weight sigma^2 / tau^2
    (sigma = residual sd of the global fit). Little data -> the ridge wins
    and a_c ~ g_c; a compound with fewer than min_stints distinct stints at the
    track is pinned to g_c. Held fixed at the global values: offsets, fuel,
    cliff/max_reliable_age (those need deep-stint data a single race does not have).

    Returns {track_id: {compound: {"age_coef", "global_age_coef", "n_stints",
    "n_laps", "pooled": True if pinned to global}}}.
    """
    glob_wear = {c: final["wear"][c][0] for c in DRY_COMPOUNDS}
    sigma2 = float(np.var(final["residual"]))
    lam = sigma2 / (tau ** 2) if tau > 0 else np.inf
    tracks = filtered["race_file"].map(track_id_of_race)
    out = {}
    order = ["HARD", "MEDIUM", "SOFT"]
    M = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=float)  # wear = M @ (A, B, C), order H, M, S
    g = np.array([glob_wear[c] for c in order])
    for tid in sorted(tracks.unique()):
        sub = filtered[tracks == tid]
        race_fuel = sub["race_file"].map(final["fuel_by_race"]).astype(float)
        y = (
            sub["lap_time_seconds"].astype(float)
            - sub["compound"].map(final["offsets"]).astype(float)
            - race_fuel * sub["lap_number"].astype(float)
        )
        groups = [sub["race_file"], sub["driver"]]
        stint_key = sub["race_file"].astype(str) + "|" + sub["driver"].astype(str) + "|" + sub["stint_id"].astype(str)
        X = pd.DataFrame(
            {c: (sub["compound"] == c).astype(float) * sub["tire_age"].astype(float) for c in DRY_COMPOUNDS},
            index=sub.index,
        )
        Xw = X - X.groupby(groups).transform("mean")
        yw = y - y.groupby(groups).transform("mean")
        n_stints = {c: int(stint_key[sub["compound"] == c].nunique()) for c in DRY_COMPOUNDS}
        n_laps = {c: int((sub["compound"] == c).sum()) for c in DRY_COMPOUNDS}
        usable = {c: bool(n_stints[c] >= min_stints and np.isfinite(lam)) for c in DRY_COMPOUNDS}

        rows, rhs = [Xw[order].values @ M], [yw.values]
        for i, c in enumerate(order):
            w = np.sqrt(lam) if usable[c] else 1e6  # 1e6 = pin to global
            rows.append(w * M[i][None, :])
            rhs.append(np.array([w * g[i]]))
        theta = lsq_linear(np.vstack(rows), np.concatenate(rhs), bounds=(0, np.inf), method="bvls").x
        wear = M @ theta
        out[tid] = {
            c: {
                "age_coef": float(wear[i]),
                "global_age_coef": float(g[i]),
                "n_stints": n_stints[c],
                "n_laps": n_laps[c],
                "pooled": not usable[c],
            }
            for i, c in enumerate(order)
        }
    return out


def model_for_track(model: dict, track_id: Optional[str]) -> dict:
    """
    The tyre model with this circuit's fitted wear slopes swapped in (T-028).
    Returns `model` itself when track_id is None / has no per-track fit, so
    callers can pass it unconditionally. Everything except each compound's
    age_coef (offsets, fuel, cliffs, max_reliable_age) is shared with the global model.
    """
    tw = (model.get("_track_wear") or {}).get(track_id) if track_id else None
    if not tw:
        return model
    out = dict(model)
    for compound, v in tw.items():
        if compound in out and isinstance(out[compound], dict):
            out[compound] = dict(out[compound], age_coef=float(v["age_coef"]))
    out["_track_id"] = track_id
    return out


def project_pace_ordering(offsets: dict) -> dict:
    """
    Project free-fit pace offsets {"SOFT", "MEDIUM", "HARD"=0} onto the physical ordering
    SOFT <= MEDIUM <= HARD(=0) (T-043). Pool-adjacent-violators from the HARD end: MEDIUM is
    clipped to <= 0, SOFT to <= the clipped MEDIUM. A pair that already satisfies the ordering is
    untouched, so a data set with a textbook ordering (2025: SOFT -0.37, MEDIUM -0.05) keeps its
    offsets; the inverted 2026 fit (SOFT +0.13, MEDIUM +0.19) projects to (0, 0).
    """
    medium = min(float(offsets.get("MEDIUM", 0.0)), 0.0)
    soft = min(float(offsets.get("SOFT", 0.0)), medium)
    return {"SOFT": soft, "MEDIUM": medium, "HARD": 0.0}


def _track_severity(model: dict, track_id: str) -> Optional[float]:
    """Track wear relative to the global fit: mean over MEDIUM/HARD of track slope / global slope (None if unfitted)."""
    tw = (model.get("_track_wear") or {}).get(track_id)
    if not tw:
        return None
    r = [tw[c]["age_coef"] / model[c]["age_coef"] for c in ("MEDIUM", "HARD") if c in tw and model[c]["age_coef"] > 0]
    return float(np.mean(r)) if r else None


def transfer_track_wear(model: dict, other: dict, track_id: str, lam: Optional[float] = None) -> Tuple[dict, Optional[float]]:
    """
    Per-track wear for a circuit this model has NO race at, borrowed from another regulation
    era's fit (T-043). `other["_track_wear"][track_id]` gives that circuit's wear relative to
    `other`'s own global slopes (its severity); the target model's global slopes are scaled by
    1 + lam * (severity - 1) per compound, then forced back onto SOFT >= MEDIUM >= HARD >= 0.

    lam (how much of an other-era circuit effect carries over) is estimated, not assumed:
    least squares through the origin of (this-era severity - 1) on (other-era severity - 1) over
    the circuits BOTH eras fitted, excluding track_id itself, clipped to [0, 1]. With ~8-10
    circuits it is a noisy number (reported in the T-043 handoff), which is why it is shrunk
    to [0, 1] and why this is off by default unless the evaluation says otherwise.

    Returns (model copy with ["_track_wear"][track_id] set, lam), or (model, None) when the
    other era has no fit for the circuit.
    """
    sev_o = _track_severity(other, track_id)
    if sev_o is None:
        return model, None
    if lam is None:
        xs, ys = [], []
        for t in (model.get("_track_wear") or {}):
            if t == track_id:
                continue
            a, b = _track_severity(other, t), _track_severity(model, t)
            if a is not None and b is not None:
                xs.append(a - 1.0)
                ys.append(b - 1.0)
        xs, ys = np.array(xs), np.array(ys)
        lam = float(np.clip((xs @ ys) / (xs @ xs), 0.0, 1.0)) if len(xs) >= 3 and (xs @ xs) > 0 else 0.0
    scale = 1.0 + lam * (sev_o - 1.0)
    h = model["HARD"]["age_coef"] * scale
    m = max(model["MEDIUM"]["age_coef"] * scale, h)
    sft = max(model["SOFT"]["age_coef"] * scale, m)
    out = dict(model)
    tw = dict(model.get("_track_wear") or {})
    tw[track_id] = {c: {"age_coef": float(v), "global_age_coef": float(model[c]["age_coef"]), "n_stints": 0, "n_laps": 0,
                        "pooled": False, "transferred_from_other_era": True}
                    for c, v in (("HARD", h), ("MEDIUM", m), ("SOFT", sft))}
    out["_track_wear"] = tw
    return out, lam


def build_tire_model(
    min_samples_per_age: int = 3,
    quadratic: bool = False,
    laps: Optional[pd.DataFrame] = None,
    eras: Optional[Iterable[str]] = DEFAULT_ERAS,
    era_weights: Optional[dict] = None,
    constrain: bool = True,
    fitted_offsets: bool = True,
    drop_red_flag_stints: bool = True,
    track_wear_tau: Optional[float] = TRACK_WEAR_TAU,
    offset_mode: str = DEFAULT_OFFSET_MODE,
) -> dict:
    """
    Full pipeline: load -> era filter -> stints -> clean -> joint fit
    (fit_joint_model) -> drop residual outliers and tyre ages with <
    min_samples_per_age laps (per compound) -> refit -> detect cliff per
    compound.

    quadratic: fit age + age^2 wear (the pre-T-002 functional form)
        instead of linear wear. Linear is the default because it predicts
        held-out races better (LORO, see module docstring). Quadratic
        disables the ordering constraints.
    laps: optional pre-loaded laps (load_all_laps() format, needs a
        race_file column) -- for leave-one-race-out validation and tests.
    eras: regulation eras to fit on (T-015). Default 2026 only: 2025 rows
        (laps given, or loaded because ERA_2025 is listed) are DROPPED
        unless the caller asks for them. If both eras are listed,
        era_weights ({era: weight}, default 1.0) down-weights an era in
        the least squares (e.g. {ERA_2025: 0.5}).
    constrain: fit with the physical ordering SOFT >= MEDIUM >= HARD >= 0
        wear and SOFT <= MEDIUM <= HARD pace (T-020, module docstring).
    fitted_offsets: use the fitted pace offsets in the simulator
        (model["_compound_offsets"]); False restores the old
        COMPOUND_PACE_OFFSET_PLACEHOLDER.
    drop_red_flag_stints: see clean_laps.
    offset_mode (T-043): "joint" = the T-020 fit (offsets and wear estimated together under the
        pace AND wear ordering constraints); "pace_ordered" = offsets are the FREE fit's, projected
        onto SOFT <= MEDIUM <= HARD(=0) only, then wear is refitted with those offsets held fixed
        under the wear ordering. See project_pace_ordering() for why.
    track_wear_tau: prior sd for the per-track wear fit (fit_track_wear) that
        fills model["_track_wear"]; None skips it. Linear-wear fits only.
        The returned model's own compound curves are always the GLOBAL ones;
        per-track values are used only through model_for_track().

    Returns e.g.:
    {
        "SOFT": {"age_coef": ..., "age2_coef": ..., "cliff_age": None,
                 "max_reliable_age": 30},
        ...,
        "_fuel_coef": -0.05,  # s of laptime per lap_number (fuel + mean track evolution)
        "_fuel_coef_by_race": {"China_2026_laps.csv": ..., ...},   # diagnostic
        "_compound_offsets": {"SOFT": ..., "MEDIUM": ..., "HARD": 0.0},  # fitted, used
        "_compound_offsets_unconstrained": {...},  # free fit, diagnostic only
        "_wear_unconstrained": {"SOFT": ..., ...},  # free linear age_coef, diagnostic only
        "_cliff_penalties": {...},           # PLACEHOLDER
        "_n_laps_fitted": 14000,
        "_eras": ["2026_regs"],
    }
    """
    eras = None if eras is None else tuple(eras)
    if laps is None:
        laps = load_all_laps(eras=eras)
    else:
        laps = filter_eras(laps, eras)
    laps = assign_stint_ids(laps)
    clean = clean_laps(laps, drop_red_flag_stints=drop_red_flag_stints)
    constrain = constrain and not quadratic

    def era_weight_of(df: pd.DataFrame) -> Optional[pd.Series]:
        if not era_weights:
            return None
        w = lap_eras(df).map(lambda e: float(era_weights.get(e, 1.0)))
        return pd.Series(w.values, index=df.index)

    first = fit_joint_model(clean, quadratic=quadratic, constrain=constrain, weights=era_weight_of(clean))
    filtered = _drop_residual_outliers(clean, first["residual"])
    counts = filtered.groupby(["compound", "tire_age"])["lap_number"].transform("size")
    filtered = filtered[counts >= min_samples_per_age]

    w_final = era_weight_of(filtered)
    if offset_mode not in ("joint", "pace_ordered"):
        raise ValueError(f"offset_mode must be 'joint' or 'pace_ordered', got {offset_mode!r}")
    if offset_mode == "pace_ordered" and not constrain:
        raise ValueError("offset_mode='pace_ordered' needs the constrained linear fit")
    if offset_mode == "joint":
        final = fit_joint_model(filtered, quadratic=quadratic, constrain=constrain, weights=w_final)
        free = final if not constrain else fit_joint_model(filtered, quadratic=quadratic, weights=w_final)
    else:
        free = fit_joint_model(filtered, quadratic=quadratic, weights=w_final)
        final = fit_joint_model(
            filtered, quadratic=quadratic, constrain=True, weights=w_final,
            fixed_offsets=project_pace_ordering(free["offsets"]),
        )

    model = {}
    # dry slicks only: rain compounds (INTERMEDIATE/WET) need their own
    # model (different degradation physics entirely) and this dataset has
    # far too few rain laps to fit anything from yet
    for compound in DRY_COMPOUNDS:
        sub = filtered[filtered["compound"] == compound]
        if sub.empty:
            continue
        age_coef, age2_coef = final["wear"][compound]
        fit = {"age_coef": age_coef, "age2_coef": age2_coef}
        cliff_age = detect_cliff(
            sub.assign(residual=final["partial_wear_residual"].loc[sub.index]), fit
        )
        model[compound] = {
            "age_coef": age_coef,
            "age2_coef": age2_coef,
            "cliff_age": cliff_age,
            "max_reliable_age": int(sub["tire_age"].max()),
        }

    model["_fuel_coef"] = final["fuel_coef"]
    model["_fuel_coef_by_race"] = {
        Path(race).name: float(v) for race, v in final["fuel_by_race"].items()
    }
    model["_compound_offsets"] = (
        dict(final["offsets"]) if fitted_offsets else dict(COMPOUND_PACE_OFFSET_PLACEHOLDER)
    )
    model["_compound_offsets_fitted"] = dict(final["offsets"])
    model["_compound_offsets_unconstrained"] = dict(free["offsets"])
    model["_wear_unconstrained"] = {c: free["wear"][c][0] for c in DRY_COMPOUNDS if c in model}
    model["_constrained"] = bool(constrain)
    model["_offset_mode"] = offset_mode
    model["_cliff_penalties"] = dict(COMPOUND_CLIFF_PENALTY_PLACEHOLDER)
    if track_wear_tau is not None and not quadratic and len(filtered):
        model["_track_wear"] = fit_track_wear(filtered, final, tau=track_wear_tau)
    model["_n_laps_fitted"] = int(len(filtered))
    model["_eras"] = sorted(set(lap_eras(filtered))) if len(filtered) else []
    return model


def fuel_effect(model: dict, lap_number: int, total_laps: int) -> float:
    """
    Seconds gained or lost to fuel load at a given lap of the race,
    anchored to zero at the final lap -- i.e. relative to a fully light,
    low-fuel car. Pairs with a base_laptime measured from a driver's
    fastest (typically late-race, low-fuel) lap.
    """
    return model["_fuel_coef"] * (lap_number - total_laps)


def predict_delta(
    model: dict, compound: str, tire_age: int, cliff_penalty_per_lap: Optional[float] = None
) -> float:
    """
    Predict seconds lost to degradation at a given tire age, relative to
    age 2. Beyond the oldest age we have enough real data for
    (max_reliable_age) -- or past the detected cliff, whichever comes
    first -- the quadratic is not trusted to extrapolate; a flat per-lap
    penalty is added on top of its value at that boundary instead.
    cliff_penalty_per_lap defaults to the compound-specific placeholder
    in model["_cliff_penalties"] (falls back to 0.3 if that key is
    missing, e.g. for hand-built test fixtures) -- pass an explicit value
    to override.

    The within-domain curve is clamped to be monotonically non-decreasing
    from age 2 onward. Some fitted curves dip below zero for a stretch
    (see this module's docstring -- fuel/track-evolution effects the
    regression doesn't fully separate from tire wear), which would
    otherwise predict a worn tire is FASTER than a fresh one of the same
    compound. That's not physically real -- a tire does not un-wear --
    so it's clamped here rather than left to quietly distort strategy
    comparisons (this is what made SOFT look implausibly strong over a
    full 50-lap stint: its curve dips negative from about age 5 to 33).
    """
    params = model[compound]
    if cliff_penalty_per_lap is None:
        cliff_penalty_per_lap = model.get("_cliff_penalties", {}).get(compound, 0.3)

    def curve(age):
        return params["age_coef"] * age + params["age2_coef"] * age**2

    boundary = params["max_reliable_age"]
    if params["cliff_age"] is not None:
        boundary = min(boundary, params["cliff_age"])

    effective_age = min(tire_age, boundary)
    if effective_age >= 2:
        peak = max(curve(age) for age in range(2, effective_age + 1))
    else:
        peak = curve(effective_age)
    delta = peak - curve(2)

    if tire_age > boundary:
        delta += cliff_penalty_per_lap * (tire_age - boundary)
    return delta


if __name__ == "__main__":
    model = build_tire_model()
    print(f"shared fuel_coef: {model['_fuel_coef']:.4f} s per lap_number (fuel + mean track evolution)")
    print("  per race: " + ", ".join(
        f"{race.split('_')[0]} {v:+.3f}" for race, v in model["_fuel_coef_by_race"].items()
    ))
    print(f"compound offsets used (PLACEHOLDER): {model['_compound_offsets']}")
    print("compound offsets fitted (diagnostic, not used): " + ", ".join(
        f"{c} {v:+.3f}" for c, v in model["_compound_offsets_fitted"].items()
    ))
    print(f"laps fitted: {model['_n_laps_fitted']}\n")
    for compound in DRY_COMPOUNDS:
        params = model[compound]
        print(
            f"{compound}: age_coef={params['age_coef']:.4f}, "
            f"age2_coef={params['age2_coef']:.5f}, cliff_age={params['cliff_age']}, "
            f"max_reliable_age={params['max_reliable_age']}"
        )
        for age in [5, 10, 15, 20, 25, 30]:
            print(f"  age {age:>2}: +{predict_delta(model, compound, age):.2f}s")
