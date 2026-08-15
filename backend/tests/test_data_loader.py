from types import SimpleNamespace

import pandas as pd

from app.core.data_loader import extract_pit_stops


def _fake_session(laps_df: pd.DataFrame) -> SimpleNamespace:
    """extract_pit_stops only touches session.laps, so a real fastf1.Session isn't needed."""
    return SimpleNamespace(laps=laps_df)


def test_extract_pit_stops_pairs_in_and_out_lap():
    laps = pd.DataFrame(
        {
            "Driver": ["1", "1", "1", "1"],
            "LapNumber": [1, 2, 3, 4],
            "PitInTime": [pd.NaT, pd.Timedelta(seconds=120), pd.NaT, pd.NaT],
            "PitOutTime": [pd.NaT, pd.NaT, pd.Timedelta(seconds=145), pd.NaT],
        }
    )

    result = extract_pit_stops(_fake_session(laps))

    assert len(result) == 1
    assert result.iloc[0]["driver"] == "1"
    assert result.iloc[0]["lap_number"] == 2
    assert result.iloc[0]["pit_duration_seconds"] == 25.0


def test_extract_pit_stops_no_stops_returns_empty_with_expected_columns():
    laps = pd.DataFrame(
        {
            "Driver": ["1", "1"],
            "LapNumber": [1, 2],
            "PitInTime": [pd.NaT, pd.NaT],
            "PitOutTime": [pd.NaT, pd.NaT],
        }
    )

    result = extract_pit_stops(_fake_session(laps))

    assert result.empty
    assert list(result.columns) == ["driver", "lap_number", "pit_duration_seconds"]


def test_extract_pit_stops_multiple_drivers():
    laps = pd.DataFrame(
        {
            "Driver": ["1", "1", "2", "2"],
            "LapNumber": [1, 2, 1, 2],
            "PitInTime": [pd.Timedelta(seconds=100), pd.NaT, pd.Timedelta(seconds=200), pd.NaT],
            "PitOutTime": [pd.NaT, pd.Timedelta(seconds=122), pd.NaT, pd.Timedelta(seconds=230)],
        }
    )

    result = extract_pit_stops(_fake_session(laps))

    assert len(result) == 2
    by_driver = result.set_index("driver")["pit_duration_seconds"]
    assert by_driver["1"] == 22.0
    assert by_driver["2"] == 30.0
