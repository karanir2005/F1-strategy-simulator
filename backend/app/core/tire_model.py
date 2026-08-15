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
"""

import glob
from pathlib import Path
from typing import Optional

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


def load_all_laps(pattern: str = f"{PROCESSED_DIR}/*_laps.csv") -> pd.DataFrame:
    """Combine every pulled race into one DataFrame, tagged by source file."""
    frames = []
    for path in glob.glob(pattern):
        df = pd.read_csv(path)
        df["race_file"] = path
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No lap files found matching {pattern}")
    return pd.concat(frames, ignore_index=True)


def assign_stint_ids(laps: pd.DataFrame) -> pd.DataFrame:
    """Tag each lap with a stint id; a new stint starts when tire_age resets to 1."""
    laps = laps.sort_values(["race_file", "driver", "lap_number"]).copy()
    is_new_stint = laps["tire_age"] == 1
    laps["stint_id"] = is_new_stint.groupby(
        [laps["race_file"], laps["driver"]]
    ).cumsum()
    return laps


def clean_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Drop out laps (tire_age == 1, cold tires), in laps (pit_in_lap == 1,
    slowed for pit entry), and any lap not run entirely under green flag
    (safety car/VSC/yellow/red mixed in -- those laps are artificially
    slow for reasons that have nothing to do with tires or fuel, and
    previously contaminated the fuel-effect fit badly enough to flip its
    sign, see fit_shared_fuel_coef's docstring). Also removes each car's
    own race-average pace so laps from different circuits/cars can be
    pooled together.
    """
    clean = laps[
        (laps["tire_age"] > 1) & (laps["pit_in_lap"] == 0) & laps["is_green_flag"]
    ].copy()
    clean["laptime_demeaned"] = clean["lap_time_seconds"] - clean.groupby(
        ["race_file", "driver"]
    )["lap_time_seconds"].transform("mean")
    return clean


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
            - predict_delta(model, row["compound"], row["tire_age"])
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


def build_tire_model(min_samples_per_age: int = 3) -> dict:
    """
    Full pipeline: load -> clean -> fit fuel+tire model -> refit on
    outlier-free data -> detect cliff, per compound. Also fits one shared
    fuel_coef across all compounds (see fit_fuel_coef_from_fresh_laps).

    Returns e.g.:
    {
        "SOFT": {"age_coef": ..., "age2_coef": ..., "cliff_age": 12},
        ...,
        "_fuel_coef": -0.045,  # seconds of laptime per lap_number, shared
        "_compound_offsets": {"SOFT": -0.5, "MEDIUM": -0.2, "HARD": 0.0},
    }
    """
    laps = load_all_laps()
    laps = assign_stint_ids(laps)
    clean = clean_laps(laps)

    model = {}
    for compound in clean["compound"].dropna().unique():
        # dry slicks only: rain compounds (INTERMEDIATE/WET) need their own
        # model (different degradation physics entirely) and this dataset
        # has far too few rain laps to fit anything from yet
        if compound not in ("SOFT", "MEDIUM", "HARD"):
            continue

        first_fit = fit_fuel_and_tire_model(clean, compound)
        residual = tire_only_residual(clean, compound, first_fit)
        filtered = remove_outliers(clean[clean["compound"] == compound], residual)

        counts = filtered.groupby("tire_age").size()
        valid_ages = counts[counts >= min_samples_per_age].index
        if valid_ages.empty:
            continue
        filtered = filtered[filtered["tire_age"].isin(valid_ages)]

        final_fit = fit_fuel_and_tire_model(filtered, compound)
        final_residual = tire_only_residual(filtered, compound, final_fit)
        filtered = filtered.assign(residual=final_residual)
        cliff_age = detect_cliff(filtered, final_fit)

        model[compound] = {
            "age_coef": final_fit["age_coef"],
            "age2_coef": final_fit["age2_coef"],
            "cliff_age": cliff_age,
            "max_reliable_age": int(valid_ages.max()),
        }

    model["_fuel_coef"] = fit_fuel_coef_from_fresh_laps(clean)
    model["_compound_offsets"] = dict(COMPOUND_PACE_OFFSET_PLACEHOLDER)
    model["_cliff_penalties"] = dict(COMPOUND_CLIFF_PENALTY_PLACEHOLDER)
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
    print(f"shared fuel_coef: {model['_fuel_coef']:.4f} s per lap_number")
    print(f"compound offsets: {model['_compound_offsets']}\n")
    for compound in ("SOFT", "MEDIUM", "HARD"):
        params = model[compound]
        print(
            f"{compound}: age_coef={params['age_coef']:.4f}, "
            f"age2_coef={params['age2_coef']:.5f}, cliff_age={params['cliff_age']}"
        )
        for age in [5, 10, 15, 20, 25, 30]:
            print(f"  age {age:>2}: +{predict_delta(model, compound, age):.2f}s")
