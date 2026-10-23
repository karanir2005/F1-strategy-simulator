"""
Safety car / VSC events for the single-car Monte Carlo (T-030).

Parameters are fitted by calibrate_safety_cars.py from the pulled per-lap data
(data/tracks/safety_car_params.json, ~40 races of 2025-26). The model is
deliberately small:

  - Event starts: each lap an SC (VSC) starts with probability
    hazard_per_lap * w(lap / race_laps), where w is the empirical start-position
    histogram blended 50/50 with a flat profile (only 26 SC and 30 VSC events,
    so the raw histogram is too lumpy to trust) and normalised to mean 1.
  - Event length: bootstrapped from the observed lengths.
  - An SC lap takes lap_multiple x the modelled green lap; a pit stop whose
    out-lap falls under SC/VSC costs pit_loss_ratio x the usual loss. Overlaps
    merge, SC taking precedence over VSC.
  - The timeline is drawn once per Monte Carlo iteration and shared by every
    strategy (common random numbers), so paired deltas stay valid.

Not modelled: reactive pit calls (plans are fixed, so an SC helps a plan only
if its stop happens to fall under it), reduced tyre wear and fuel burn under SC,
bunching/restarts, red flags, and track-specific SC rates (too few races per track).
"""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import numpy as np

PARAMS_PATH = Path(__file__).resolve().parents[3] / "data" / "tracks" / "safety_car_params.json"
UNIFORM_BLEND = 0.5


@dataclass
class SafetyCarParams:
    sc_hazard_per_lap: float
    vsc_hazard_per_lap: float
    sc_start_hist: Tuple[float, ...]  # counts per tenth of the race
    vsc_start_hist: Tuple[float, ...]
    sc_lengths: Tuple[int, ...]
    vsc_lengths: Tuple[int, ...]
    sc_lap_multiple: float
    vsc_lap_multiple: float
    sc_pit_loss_ratio: float
    vsc_pit_loss_ratio: float


@lru_cache(maxsize=1)
def load_params() -> SafetyCarParams:
    d = json.loads(PARAMS_PATH.read_text())
    return SafetyCarParams(
        sc_hazard_per_lap=d["sc"]["hazard_per_lap"],
        vsc_hazard_per_lap=d["vsc"]["hazard_per_lap"],
        sc_start_hist=tuple(d["sc"]["start_hist_decile_counts"]),
        vsc_start_hist=tuple(d["vsc"]["start_hist_decile_counts"]),
        sc_lengths=tuple(d["sc"]["length_laps_values"]),
        vsc_lengths=tuple(d["vsc"]["length_laps_values"]),
        sc_lap_multiple=d["sc"]["lap_multiple_median"],
        vsc_lap_multiple=d["vsc"]["lap_multiple_median"],
        sc_pit_loss_ratio=d["pit_loss_ratio"]["sc"],
        vsc_pit_loss_ratio=d["pit_loss_ratio"]["vsc"],
    )


def _draw_mask(rng, n_sims, n_laps, hazard, hist, lengths):
    h = np.asarray(hist, dtype=float)
    shape = UNIFORM_BLEND + (1 - UNIFORM_BLEND) * h / h.mean()
    frac = (np.arange(n_laps) + 0.5) / n_laps
    w = shape[np.minimum((frac * len(h)).astype(int), len(h) - 1)]
    starts = rng.random((n_sims, n_laps)) < np.clip(hazard * w, 0, 1)
    lens = rng.choice(np.asarray(lengths), size=(n_sims, n_laps))
    delta = np.zeros((n_sims, n_laps + 1), dtype=np.int16)
    rows, cols = np.nonzero(starts)
    np.add.at(delta, (rows, cols), 1)
    np.add.at(delta, (rows, np.minimum(cols + lens[rows, cols], n_laps)), -1)
    return np.cumsum(delta[:, :n_laps], axis=1) > 0


def draw_timelines(rng: np.random.Generator, n_sims: int, n_laps: int, params: SafetyCarParams):
    """(sc_mask, vsc_mask), each (n_sims, n_laps) bool; vsc excludes SC laps."""
    sc = _draw_mask(rng, n_sims, n_laps, params.sc_hazard_per_lap, params.sc_start_hist, params.sc_lengths)
    vsc = _draw_mask(rng, n_sims, n_laps, params.vsc_hazard_per_lap, params.vsc_start_hist, params.vsc_lengths)
    return sc, vsc & ~sc


def strategy_extra(sc, vsc, lap_times, pit_laps, pit_loss, params: SafetyCarParams) -> np.ndarray:
    """Per-simulation seconds added to a strategy's total by the shared SC/VSC timeline."""
    n = len(lap_times)
    green = np.asarray(lap_times, dtype=float).copy()
    pl = np.asarray(pit_laps, dtype=int) - 1
    green[pl] -= pit_loss  # slowdown applies to the lap itself, pit loss is scaled separately
    extra = sc[:, :n] @ ((params.sc_lap_multiple - 1) * green)
    extra += vsc[:, :n] @ ((params.vsc_lap_multiple - 1) * green)
    for p in pl:
        extra -= (1 - params.sc_pit_loss_ratio) * pit_loss * sc[:, p]
        extra -= (1 - params.vsc_pit_loss_ratio) * pit_loss * vsc[:, p]
    return extra
