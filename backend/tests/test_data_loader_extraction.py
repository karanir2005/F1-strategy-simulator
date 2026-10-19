"""
T-002: extract_laps_df / sidecar extractors on hand-built sessions (no
FastF1 load, no network). Each test builds the minimal session.laps /
results / race_control_messages / weather_data the extractor reads.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.core.data_loader import (
    _overtake_disabled_laps,
    extract_laps_df,
    extract_results_df,
    pit_stops_from_laps,
)


def _td(seconds):
    return pd.NaT if seconds is None else pd.Timedelta(seconds=seconds)


def _lap_rows(driver, laps, start_time, lap_time=90.0, team="T"):
    """
    laps: list of dicts with keys stint, compound, age, and optionally
    pit_in, pit_out, ts (TrackStatus), lap_time (None -> NaN LapTime, the
    lap still takes 120 s of session time), position, generated, fresh.
    PitOutTime is 5 s into the lap, PitInTime 5 s before its end.
    """
    rows, t = [], start_time
    for i, spec in enumerate(laps, start=1):
        lt = spec.get("lap_time", lap_time)
        lap_start = t
        t = t + (lt if lt is not None else 120.0)
        rows.append(
            {
                "Time": _td(t),
                "Driver": driver,
                "DriverNumber": "99",
                "LapTime": _td(lt),
                "LapNumber": float(i),
                "Stint": float(spec["stint"]),
                "PitOutTime": _td(lap_start + 5) if spec.get("pit_out") else pd.NaT,
                "PitInTime": _td(t - 5) if spec.get("pit_in") else pd.NaT,
                "Sector1Time": _td(None if i == 1 else 30.0),
                "Sector2Time": _td(30.0),
                "Sector3Time": _td(30.0),
                "Compound": spec["compound"],
                "TyreLife": spec["age"],
                "FreshTyre": spec.get("fresh", spec["age"] == 1),
                "Team": team,
                "LapStartTime": _td(lap_start),
                "TrackStatus": spec.get("ts", "1"),
                "Position": spec.get("position", np.nan),
                "Deleted": False,
                "FastF1Generated": spec.get("generated", False),
                "IsAccurate": True,
            }
        )
    return rows


def _session(laps_rows, results=None, rcm=None, weather=None):
    laps = pd.DataFrame(laps_rows)
    if results is None:
        results = pd.DataFrame({"Abbreviation": laps["Driver"].unique(), "Status": "Finished"})
    if rcm is None:
        rcm = pd.DataFrame(columns=["Time", "Message", "Lap", "Category"])
    if weather is None:
        weather = pd.DataFrame(
            {
                "Time": [_td(0), _td(10_000)],
                "TrackTemp": [40.0, 41.0],
                "AirTemp": [25.0, 26.0],
                "Rainfall": [False, False],
            }
        )
    return SimpleNamespace(laps=laps, results=results, race_control_messages=rcm, weather_data=weather)


def _stint_spec(n, stint, compound, start_age, **first_lap_extra):
    specs = [{"stint": stint, "compound": compound, "age": float(start_age + k)} for k in range(n)]
    specs[0].update(first_lap_extra)
    return specs


def test_used_set_stop_starts_new_tyre_stint_with_start_age():
    # MEDIUM new x3, pit -> SOFT used (age 5) x3: the old tire_age==1 rule merged these
    specs = _stint_spec(3, 1, "MEDIUM", 1) + _stint_spec(3, 2, "SOFT", 5, pit_out=True)
    specs[2]["pit_in"] = True
    df = extract_laps_df(_session(_lap_rows("AAA", specs, 0.0)), compound_map={})
    assert df["tyre_stint"].tolist() == [1, 1, 1, 2, 2, 2]
    assert df["stint_start_age"].tolist() == [1, 1, 1, 5, 5, 5]
    assert df["is_tyre_change"].tolist() == [False, False, False, True, False, False]
    assert df["pit_out_lap"].tolist() == [0, 0, 0, 1, 0, 0]
    assert df["stint_id"].iloc[-1] == "AAA_2"


def test_same_tyre_pit_pass_is_not_a_tyre_change():
    # drive-through: FastF1 Stint increments but compound and TyreLife continue
    specs = _stint_spec(3, 1, "HARD", 1) + [{"stint": 2, "compound": "HARD", "age": 4.0, "pit_out": True}]
    specs[2]["pit_in"] = True
    df = extract_laps_df(_session(_lap_rows("AAA", specs, 0.0)), compound_map={})
    assert df["tyre_stint"].tolist() == [1, 1, 1, 1]
    assert df["ff1_stint"].tolist() == [1, 1, 1, 2]
    assert df["is_pit_pass_no_change"].tolist() == [False, False, False, True]
    assert not df["is_tyre_change"].any()


def test_same_compound_new_set_is_a_tyre_change():
    # MEDIUM -> fresh MEDIUM: compound unchanged but TyreLife resets
    specs = _stint_spec(3, 1, "MEDIUM", 1) + _stint_spec(2, 2, "MEDIUM", 1, pit_out=True)
    specs[2]["pit_in"] = True
    df = extract_laps_df(_session(_lap_rows("AAA", specs, 0.0)), compound_map={})
    assert df["tyre_stint"].tolist() == [1, 1, 1, 2, 2]


def test_lap_one_used_set_start_age():
    df = extract_laps_df(_session(_lap_rows("AAA", _stint_spec(3, 1, "SOFT", 4), 0.0)), compound_map={})
    assert df["stint_start_age"].tolist() == [4, 4, 4]
    assert df["tyre_stint"].tolist() == [1, 1, 1]


def test_rows_kept_and_dropped():
    specs = _stint_spec(4, 1, "MEDIUM", 1)
    specs[1]["lap_time"] = None  # e.g. an SC lap with no LapTime -> KEPT
    specs[3]["generated"] = True  # synthetic retiree row -> DROPPED
    dns = [{"stint": 1, "compound": "SOFT", "age": 1.0, "lap_time": None}]
    rows = _lap_rows("AAA", specs, 0.0) + _lap_rows("DNS", dns, 0.0)
    results = pd.DataFrame({"Abbreviation": ["AAA", "DNS"], "Status": ["Finished", "Did not start"]})
    df = extract_laps_df(_session(rows, results=results), compound_map={})
    assert df["driver"].unique().tolist() == ["AAA"]
    assert df["lap_number"].tolist() == [1, 2, 3]
    assert np.isnan(df.loc[1, "lap_time_seconds"])
    assert df["lap_end_s"].notna().all()


def test_track_status_decoded_per_character():
    codes = ["1", "12", "4", "671", "15", "21"]
    specs = [{"stint": 1, "compound": "HARD", "age": float(i + 1), "ts": c} for i, c in enumerate(codes)]
    df = extract_laps_df(_session(_lap_rows("AAA", specs, 0.0)), compound_map={})
    assert df["is_green_flag"].tolist() == [True, False, False, False, False, False]
    assert df["has_yellow"].tolist() == [False, True, False, False, False, True]
    assert df["has_sc"].tolist() == [False, False, True, False, False, False]
    assert df["has_vsc"].tolist() == [False, False, False, True, False, False]
    assert df["has_red"].tolist() == [False, False, False, False, True, False]
    assert df["is_neutralised"].tolist() == [False, False, True, True, True, False]
    # lap 1 and SC/red laps have Overtake off
    assert df["overtake_enabled"].tolist() == [False, True, False, True, False, True]


def test_tyre_imputation_partial_and_whole_stint():
    specs = (
        _stint_spec(4, 1, "MEDIUM", 1)
        + _stint_spec(3, 2, "HARD", 1, pit_out=True)
        + _stint_spec(2, 3, "SOFT", 1, pit_out=True)
    )
    specs[3]["pit_in"] = True
    specs[6]["pit_in"] = True
    specs[2]["compound"], specs[2]["age"] = None, np.nan  # gap inside a known stint -> imputed
    for k in (4, 5, 6):  # whole stint 2 unknown (the Hungary 2026 PER case) -> stays missing
        specs[k]["compound"], specs[k]["age"] = None, np.nan
    df = extract_laps_df(_session(_lap_rows("AAA", specs, 0.0)), compound_map={})
    assert df.loc[2, "compound"] == "MEDIUM" and df.loc[2, "tire_age"] == 3
    assert df.loc[2, "tyre_data_imputed"] and not df.loc[2, "tyre_data_missing"]
    assert df.loc[4:6, "tyre_data_missing"].all()
    assert df.loc[4:6, "compound"].isna().all()
    # the unknown stint stays its own stint (not merged with the MEDIUM before or SOFT after)
    assert df["tyre_stint"].tolist() == [1, 1, 1, 1, 2, 2, 2, 3, 3]


def test_gaps_from_end_of_lap_time():
    a = _lap_rows(
        "AAA", [{"stint": 1, "compound": "HARD", "age": float(i), "position": 1} for i in (1, 2)], 0.0, lap_time=90.0
    )
    b = _lap_rows(
        "BBB", [{"stint": 1, "compound": "HARD", "age": float(i), "position": 2} for i in (1, 2)], 0.0, lap_time=91.5
    )
    df = extract_laps_df(_session(a + b), compound_map={}).set_index(["driver", "lap_number"])
    assert np.isnan(df.loc[("AAA", 2), "gap_ahead_s"])
    assert df.loc[("BBB", 1), "gap_ahead_s"] == pytest.approx(1.5)
    assert df.loc[("BBB", 2), "gap_ahead_s"] == pytest.approx(3.0)
    assert df.loc[("BBB", 2), "car_ahead"] == "AAA"
    assert df.loc[("BBB", 2), "gap_leader_s"] == pytest.approx(3.0)
    # on-road: BBB lap 1 (t=91.5) crosses the line right behind AAA lap 1 (t=90)
    assert df.loc[("BBB", 1), "road_car_ahead"] == "AAA"
    assert df.loc[("BBB", 1), "road_gap_ahead_s"] == pytest.approx(1.5)
    assert df.loc[("BBB", 1), "road_ahead_lap_diff"] == 0


def test_pit_columns_and_pit_stop_table():
    specs = _stint_spec(3, 1, "MEDIUM", 1) + _stint_spec(2, 2, "HARD", 1, pit_out=True)
    specs[2]["pit_in"] = True
    specs[2]["ts"] = "14"  # stopped under SC
    df = extract_laps_df(_session(_lap_rows("AAA", specs, 0.0)), compound_map={"MEDIUM": "C3", "HARD": "C2"})
    # in-lap ends at t=270 (PitIn at 265); out-lap starts at 270 (PitOut at 275)
    assert df.loc[3, "pit_lane_s"] == pytest.approx(10.0)
    assert df.loc[2, "pit_under_sc"] and not df.loc[3, "pit_under_sc"]
    assert df["compound_c"].tolist() == ["C3", "C3", "C3", "C2", "C2"]
    stops = pit_stops_from_laps(df)
    assert len(stops) == 1
    row = stops.iloc[0]
    assert (row["in_lap"], row["out_lap"]) == (3, 4)
    assert row["is_tyre_change"] and row["pit_under_sc"]
    assert (row["compound_before"], row["compound_after"]) == ("MEDIUM", "HARD")


def test_compound_c_passes_wet_tyres_through():
    df = extract_laps_df(_session(_lap_rows("AAA", _stint_spec(2, 1, "INTERMEDIATE", 1), 0.0)), compound_map={"SOFT": "C5"})
    assert df["compound_c"].tolist() == ["INTER", "INTER"]


def test_weather_is_sample_at_or_before_lap_end():
    weather = pd.DataFrame(
        {
            "Time": [_td(0), _td(100), _td(200)],
            "TrackTemp": [30.0, 35.0, 40.0],
            "AirTemp": [20.0, 21.0, 22.0],
            "Rainfall": [False, False, True],
        }
    )
    # laps end at 90, 180, 270
    df = extract_laps_df(_session(_lap_rows("AAA", _stint_spec(3, 1, "HARD", 1), 0.0), weather=weather), compound_map={})
    assert df["track_temp_c"].tolist() == [30.0, 35.0, 40.0]


def test_overtake_disabled_laps_from_race_control():
    rcm = pd.DataFrame(
        {
            "Time": pd.to_datetime(["2026-01-01 10:00", "2026-01-01 10:05", "2026-01-01 10:40", "2026-01-01 10:50"]),
            "Message": ["OVERTAKE DISABLED", "OVERTAKE ENABLED", "OVERTAKE DISABLED", "OVERTAKE ENABLED"],
            "Lap": [1, 1, 14, 17],
        }
    )
    assert _overtake_disabled_laps(rcm, 30) == {1, 14, 15, 16, 17}
    # never re-enabled -> disabled to the end
    assert _overtake_disabled_laps(rcm.iloc[:3], 16) == {1, 14, 15, 16}


def test_extract_results_gap_semantics():
    results = pd.DataFrame(
        {
            "Abbreviation": ["WIN", "P2", "LAP", "DNF"],
            "DriverNumber": ["1", "2", "3", "4"],
            "TeamName": ["A", "B", "C", "D"],
            "GridPosition": [2.0, 1.0, 3.0, 4.0],
            "Position": [1.0, 2.0, 3.0, 4.0],
            "ClassifiedPosition": ["1", "2", "3", "R"],
            "Status": ["Finished", "Finished", "Lapped", "Retired"],
            "Laps": [50.0, 50.0, 49.0, 10.0],
            "Points": [25.0, 18.0, 15.0, 0.0],
            # winner: total; other lead-lap cars: gap to winner; lapped: not a usable gap
            "Time": [_td(5000), _td(5.5), _td(4.4), pd.NaT],
            "Q1": [pd.NaT] * 4,
            "Q2": [pd.NaT] * 4,
            "Q3": [pd.NaT] * 4,
        }
    )
    df = extract_results_df(SimpleNamespace(results=results)).set_index("driver")
    assert df.loc["WIN", "gap_to_winner_s"] == 0.0
    assert df.loc["WIN", "race_time_s"] == pytest.approx(5000.0)
    assert df.loc["P2", "race_time_s"] == pytest.approx(5005.5)
    assert np.isnan(df.loc["LAP", "gap_to_winner_s"])
    assert np.isnan(df.loc["DNF", "race_time_s"])
