"""T-002: validate_replay.reconstruct_strategy / actual_race_time (repo-root script)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from validate_replay import actual_race_time, reconstruct_strategy  # noqa: E402


def _driver_laps():
    # MEDIUM new L1-3, drive-through on L4 (same tyre), SOFT used age 5 L5-7
    return pd.DataFrame(
        {
            "lap_number": [1, 2, 3, 4, 5, 6, 7],
            "lap_time_seconds": [95.0, 90.0, np.nan, 110.0, 112.0, 89.0, 89.0],
            "lap_start_s": [0.0, 95.0, 185.0, 305.0, 415.0, 527.0, 616.0],
            "lap_end_s": [95.0, 185.0, 305.0, 415.0, 527.0, 616.0, 705.0],
            "compound": ["MEDIUM"] * 4 + ["SOFT"] * 3,
            "tire_age": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "tyre_stint": [1, 1, 1, 1, 2, 2, 2],
            "stint_start_age": [1, 1, 1, 1, 5, 5, 5],
            "ff1_stint": [1, 1, 1, 2, 3, 3, 3],
        }
    )


def test_reconstruct_strategy_sets_start_age_and_ignores_pit_passes():
    strategy = reconstruct_strategy(_driver_laps())
    assert [(s.compound, s.laps, s.start_age) for s in strategy] == [("MEDIUM", 4, 1), ("SOFT", 3, 5)]


def test_reconstruct_strategy_counts_laps_without_lap_time():
    # lap 3 has no LapTime (e.g. SC) but was still driven
    assert sum(s.laps for s in reconstruct_strategy(_driver_laps())) == 7


def test_reconstruct_strategy_legacy_csv_fallback():
    legacy = _driver_laps().drop(columns=["tyre_stint", "stint_start_age", "ff1_stint"])
    strategy = reconstruct_strategy(legacy)
    # legacy rule can't see the used-set stop -- documented limitation
    assert [(s.compound, s.laps, s.start_age) for s in strategy] == [("MEDIUM", 7, 1)]


def test_reconstruct_strategy_rejects_unknown_tyres():
    laps = _driver_laps()
    laps.loc[laps["tyre_stint"] == 2, "compound"] = np.nan
    with pytest.raises(ValueError, match="no tyre data"):
        reconstruct_strategy(laps)


def test_actual_race_time_uses_timing_line_not_lap_time_sum():
    laps = _driver_laps()
    # 705 s end of lap 7; lap 1 ends at 95 s with LapTime 95 -> 705 s
    assert actual_race_time(laps) == pytest.approx(705.0)
    assert laps["lap_time_seconds"].sum() < 705.0  # the old sum silently dropped lap 3
