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

Compound pace offsets stay on COMPOUND_PACE_OFFSET_PLACEHOLDER: the joint
fit still says SOFT (+0.38 s) and MEDIUM (+0.18 s) are SLOWER than HARD,
and keying on Pirelli C-number instead of the S/M/H label doesn't help
(C1 -0.47, C2 -0.28, C4 +0.18, C5 +0.25 s vs C3). Since that survives
the stint fix, out-lap/restart cleaning, per-race fuel and C-number
keying, it is not a data-cleaning artefact; candidates are traffic /
stint-situation selection and 2026 tyre/energy management. The fitted
values are exposed as model["_compound_offsets_fitted"] (diagnostic only).
"""

import glob
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# resolved from this file's location, not the caller's cwd -- otherwise
# this breaks the moment something imports tire_model from a process
# launched with a different working directory (e.g. `uvicorn` started
# from backend/ instead of the repo root)
PROCESSED_DIR = str(Path(__file__).resolve().parents[3] / "data" / "processed")

# Placeholder, not fitted from data -- see fit_compound_pace_offsets'
# docstring for why. Seconds of laptime relative to HARD (negative =
# faster); rough F1 ballpark, varies significantly by track in reality.
COMPOUND_PACE_OFFSET_PLACEHOLDER = {"SOFT": -0.5, "MEDIUM": -0.2, "HARD": 0.0}

# Placeholder, not fitted from data. A single global 0.3 s/lap cliff
# penalty (the old default) understates how differently compounds fall
# off: our own data barely has any laps run deep past a soft's natural
# life (see max_reliable_age's docstring -- SOFT samples drop to n=2 past
# age ~25), so there's nothing real to fit a per-compound severity from
# yet. These follow the general F1 ordering (softer = falls off harder
# once truly gone) as a ballpark, not a measurement.
COMPOUND_CLIFF_PENALTY_PLACEHOLDER = {"SOFT": 1.0, "MEDIUM": 0.5, "HARD": 0.2}


DRY_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")


def load_all_laps(pattern: str = f"{PROCESSED_DIR}/*_laps.csv") -> pd.DataFrame:
    """Combine every pulled race into one DataFrame, tagged by source file."""
    frames = []
    for path in sorted(glob.glob(pattern)):
        df = pd.read_csv(path)
        df["race_file"] = path
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No lap files found matching {pattern}")
    return pd.concat(frames, ignore_index=True)


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


def clean_laps(laps: pd.DataFrame) -> pd.DataFrame:
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
    else:
        keep &= laps["tire_age"] > 1
    clean = laps[keep].copy()
    clean["laptime_demeaned"] = clean["lap_time_seconds"] - clean.groupby(
        ["race_file", "driver"]
    )["lap_time_seconds"].transform("mean")
    return clean


def _joint_design(clean: pd.DataFrame, quadratic: bool) -> Tuple[pd.DataFrame, list]:
    """Design matrix for fit_joint_model (before the within transformation)."""
    X = pd.DataFrame(index=clean.index)
    for compound in ("SOFT", "MEDIUM"):  # HARD is the reference level
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


def fit_joint_model(clean: pd.DataFrame, quadratic: bool = False) -> dict:
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

    Returns {"coef": {name: value}, "fuel_coef": lap-weighted mean of the
    per-race lap_number slopes, "fuel_by_race": {...}, "offsets": {...},
    "wear": {compound: (age_coef, age2_coef)}, "residual": Series of the
    full-model residual per lap, "partial_wear_residual": Series of
    y - FE - offsets - fuel (i.e. what's left for the wear curve)}.
    """
    X, cols = _joint_design(clean, quadratic)
    groups = [clean["race_file"], clean["driver"]]
    y = clean["lap_time_seconds"].astype(float)
    Xw = X - X.groupby(groups).transform("mean")
    yw = y - y.groupby(groups).transform("mean")
    beta, *_ = np.linalg.lstsq(Xw.values, yw.values, rcond=None)
    coef = dict(zip(cols, beta))

    fitted = X.values @ beta
    fe = (y - fitted).groupby(groups).transform("mean")
    residual = y - fitted - fe
    wear_cols = [c for c in cols if c.startswith("age")]
    wear_part = X[wear_cols].values @ np.array([coef[c] for c in wear_cols])
    partial_wear_residual = residual + wear_part

    fuel_by_race = {c[len("fuel_"):]: coef[c] for c in cols if c.startswith("fuel_")}
    race_laps = clean["race_file"].value_counts()
    fuel_coef = float(
        np.average(list(fuel_by_race.values()), weights=[race_laps[r] for r in fuel_by_race])
    )
    offsets = {"HARD": 0.0}
    for compound in ("SOFT", "MEDIUM"):
        offsets[compound] = float(coef.get(f"off_{compound}", 0.0))
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


def build_tire_model(
    min_samples_per_age: int = 3,
    quadratic: bool = False,
    laps: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Full pipeline: load -> stints -> clean -> joint fit (fit_joint_model)
    -> drop residual outliers and tyre ages with < min_samples_per_age
    laps (per compound) -> refit -> detect cliff per compound.

    quadratic: fit age + age^2 wear (the pre-T-002 functional form)
        instead of linear wear. Linear is the default because it predicts
        held-out races better (LORO, see module docstring).
    laps: optional pre-loaded laps (load_all_laps() format, needs a
        race_file column) -- for leave-one-race-out validation and tests.

    Returns e.g.:
    {
        "SOFT": {"age_coef": ..., "age2_coef": ..., "cliff_age": None,
                 "max_reliable_age": 30},
        ...,
        "_fuel_coef": -0.045,  # s of laptime per lap_number (fuel + mean track evolution)
        "_fuel_coef_by_race": {"China_2026_laps.csv": ..., ...},   # diagnostic
        "_compound_offsets": {"SOFT": -0.5, "MEDIUM": -0.2, "HARD": 0.0},  # PLACEHOLDER, used
        "_compound_offsets_fitted": {...},   # joint-fit estimate, diagnostic only (inverted)
        "_cliff_penalties": {...},           # PLACEHOLDER
        "_n_laps_fitted": 8700,
    }
    """
    if laps is None:
        laps = load_all_laps()
    laps = assign_stint_ids(laps)
    clean = clean_laps(laps)

    first = fit_joint_model(clean, quadratic=quadratic)
    filtered = _drop_residual_outliers(clean, first["residual"])
    counts = filtered.groupby(["compound", "tire_age"])["lap_number"].transform("size")
    filtered = filtered[counts >= min_samples_per_age]

    final = fit_joint_model(filtered, quadratic=quadratic)

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
    model["_compound_offsets"] = dict(COMPOUND_PACE_OFFSET_PLACEHOLDER)
    model["_compound_offsets_fitted"] = dict(final["offsets"])
    model["_cliff_penalties"] = dict(COMPOUND_CLIFF_PENALTY_PLACEHOLDER)
    model["_n_laps_fitted"] = int(len(filtered))
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
