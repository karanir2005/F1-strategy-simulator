"""T-009: OpenF1 -> laps CSV loader. Offline: synthetic raw sessions, plus
one real-data check that skips if the raw cache isn't present."""
import os
import warnings

import numpy as np
import pandas as pd
import pytest

from app.core import openf1_loader as o

warnings.filterwarnings("ignore", category=FutureWarning)

T0 = pd.Timestamp("2026-01-01T12:00:00Z")


def iso(sec):
    return (T0 + pd.Timedelta(seconds=sec)).isoformat()


def make_raw(n_laps=10, pit_lap=5, stint_start_on_inlap=False, trailing_none=False,
             sc=None, extra_post_race=False):
    """Two drivers (#1 AAA, #2 BBB), 100 s laps. AAA pits at `pit_lap`."""
    drivers = [
        {"driver_number": 1, "name_acronym": "AAA", "team_name": "T1"},
        {"driver_number": 2, "name_acronym": "BBB", "team_name": "T2"},
    ]
    laps = []
    for dn in (1, 2):
        for n in range(1, n_laps + 1):
            laps.append({
                "driver_number": dn, "lap_number": n, "date_start": iso((n - 1) * 100 + dn * 0.3),
                "lap_duration": None if (n == 1 and dn == 1) else 100.0,
                "duration_sector_1": 30.0, "duration_sector_2": 40.0, "duration_sector_3": 30.0,
                "is_pit_out_lap": dn == 1 and n == pit_lap + 1,
            })
        if extra_post_race and dn == 1:
            laps.append({"driver_number": 1, "lap_number": n_laps + 1,
                         "date_start": iso(n_laps * 100 + 0.3), "lap_duration": None,
                         "duration_sector_1": None, "duration_sector_2": None, "duration_sector_3": None,
                         "is_pit_out_lap": False})
    ls2 = pit_lap if stint_start_on_inlap else pit_lap + 1
    stints = [
        {"driver_number": 1, "stint_number": 1, "lap_start": 1, "lap_end": ls2 - 1, "compound": "MEDIUM",
         "tyre_age_at_start": 0},
        {"driver_number": 1, "stint_number": 2, "lap_start": ls2, "lap_end": n_laps, "compound": "HARD",
         "tyre_age_at_start": 0},
        {"driver_number": 2, "stint_number": 1, "lap_start": 1, "lap_end": n_laps, "compound": "HARD",
         "tyre_age_at_start": 3},
    ]
    if trailing_none:
        stints[1]["lap_end"] = n_laps - 1
        stints.append({"driver_number": 1, "stint_number": 3, "lap_start": n_laps, "lap_end": n_laps,
                       "compound": None, "tyre_age_at_start": 0})
    pit = [{"driver_number": 1, "lap_number": pit_lap, "lane_duration": 22.5, "pit_duration": 22.5,
            "stop_duration": 2.4, "date": iso(pit_lap * 100)}]
    rc = [{"date": iso(-60), "lap_number": 1, "category": "Flag", "flag": "GREEN", "scope": "Track",
           "sector": None, "driver_number": None, "message": "GREEN LIGHT - PIT EXIT OPEN"}]
    if sc:
        a, b = sc
        rc += [{"date": iso(a), "lap_number": 3, "category": "SafetyCar", "flag": None, "scope": None,
                "sector": None, "driver_number": None, "message": "SAFETY CAR DEPLOYED"},
               {"date": iso(b), "lap_number": 4, "category": "Flag", "flag": "CLEAR", "scope": "Track",
                "sector": None, "driver_number": None, "message": "TRACK CLEAR"}]
    position = [{"date": iso(-10), "driver_number": 1, "position": 1},
                {"date": iso(-10), "driver_number": 2, "position": 2}]
    weather = [{"date": iso(-30), "track_temperature": 30.0, "air_temperature": 20.0, "rainfall": 0,
                "humidity": 50.0, "pressure": 1010.0, "wind_direction": 10, "wind_speed": 2.0}]
    return {"drivers": drivers, "laps": laps, "stints": stints, "pit": pit, "race_control": rc,
            "weather": weather, "position": position, "intervals": [], "session_result": [],
            "starting_grid": []}


def conv(raw, **kw):
    laps, aux = o.convert_laps(raw, 2026, None, **kw)
    return laps.set_index(["driver", "lap_number"]), aux


def test_tire_age_convention_matches_fastf1():
    laps, _ = conv(make_raw())
    a = laps.loc["AAA"]
    assert a.loc[1, "tire_age"] == 1 and a.loc[5, "tire_age"] == 5      # new set: 0 -> 1
    assert a.loc[6, "tire_age"] == 1 and a.loc[6, "compound"] == "HARD"
    assert laps.loc[("BBB", 1), "tire_age"] == 4 and laps.loc[("BBB", 1), "stint_start_age"] == 4


def test_real_tyre_change_detected_once():
    laps, _ = conv(make_raw())
    a = laps.loc["AAA"]
    assert list(a.index[a["is_tyre_change"]]) == [6]
    assert a["tyre_stint"].tolist() == [1] * 5 + [2] * 5
    assert a.loc[5, "pit_in_lap"] == 1 and a.loc[6, "pit_out_lap"] == 1
    assert a.loc[6, "pit_lane_s"] == pytest.approx(22.5) and np.isnan(a.loc[5, "pit_lane_s"])


def test_stint_starting_on_inlap_is_shifted_to_outlap():
    # OpenF1 quirk (Canada/Monaco/...): stint 2 lap_start == the pit in-lap.
    laps, _ = conv(make_raw(stint_start_on_inlap=True))
    a = laps.loc["AAA"]
    assert a.loc[5, "compound"] == "MEDIUM" and a.loc[5, "tire_age"] == 5
    assert a.loc[6, "compound"] == "HARD" and a.loc[6, "tire_age"] == 1
    assert list(a.index[a["is_tyre_change"]]) == [6]


def test_trailing_null_compound_stint_is_merged_not_a_phantom_change():
    laps, _ = conv(make_raw(trailing_none=True))
    a = laps.loc["AAA"]
    assert a["tyre_stint"].max() == 2
    assert a.loc[10, "compound"] == "HARD" and a.loc[10, "tire_age"] == 5


def test_post_race_untimed_row_trimmed_with_jolpica_laps():
    raw = make_raw(extra_post_race=True)
    full, _ = conv(raw)
    assert 11 in full.loc["AAA"].index
    trimmed, aux = conv(raw, result_laps={"AAA": (10, "Finished"), "BBB": (10, "Finished")})
    assert 11 not in trimmed.loc["AAA"].index and aux["trimmed"] == 1


def test_lap1_duration_filled_from_next_lap_start():
    laps, _ = conv(make_raw())
    assert laps.loc[("AAA", 1), "lap_time_seconds"] == pytest.approx(100.0, abs=0.01)


def test_track_status_sc_window_and_flags():
    # SC from t=210 to t=450 => laps 3 (200-300), 4, 5 (400-500) touched
    laps, _ = conv(make_raw(sc=(210, 450)))
    a = laps.loc["AAA"]
    assert a.loc[2, "track_status"] == "1" and bool(a.loc[2, "is_green_flag"])
    assert a.loc[3, "track_status"] == "14" and bool(a.loc[3, "has_sc"])
    assert a.loc[4, "track_status"] == "4"
    assert a.loc[5, "track_status"] == "41"
    assert a.loc[6, "track_status"] == "1" and not bool(a.loc[6, "is_neutralised"])
    # pit under SC would be flagged on the in-lap only if the in-lap is an SC lap
    assert not bool(a.loc[5, "pit_under_vsc"])


def test_columns_match_fastf1_schema_plus_season_era_source():
    laps, _ = o.convert_laps(make_raw(), 2026, None)
    path = os.path.join(o.REPO_ROOT, "data", "processed", "Japan_2026_laps.csv")
    if not os.path.exists(path):
        pytest.skip("processed CSVs not present")
    ff1_cols = list(pd.read_csv(path, nrows=1).columns)
    cols = list(laps.columns)
    assert cols[: len(ff1_cols)] == ff1_cols
    assert cols[len(ff1_cols):] == ["season", "era", "source"]
    assert set(laps["era"]) == {"2026_regs"} and set(laps["source"]) == {"openf1"} and set(laps["season"]) == {2026}


def test_parse_lap_time_strings_and_quali():
    assert o._lap_str_to_s("1:30.035") == pytest.approx(90.035)
    assert o._lap_str_to_s("59.5") == pytest.approx(59.5)
    assert np.isnan(o._lap_str_to_s("")) and np.isnan(o._lap_str_to_s(None))
    jq = {"MRData": {"RaceTable": {"Races": [{"QualifyingResults": [
        {"number": "12", "position": "1", "Driver": {"code": "ANT"}, "Constructor": {"name": "Mercedes"},
         "Q1": "1:30.035", "Q2": "1:29.048", "Q3": "1:28.778"},
        {"number": "99", "position": "20", "Driver": {"code": "ZZZ"}, "Constructor": {"name": "X"},
         "Q1": "1:35.000"}]}]}}}
    q = o.convert_quali(jq, 2025)
    assert q.loc[0, "q3_s"] == pytest.approx(88.778) and np.isnan(q.loc[1, "q2_s"])
    assert set(q["era"]) == {"2025_regs"}


def test_results_gap_only_for_finished_cars():
    def res(pos, code, status, millis, grid):
        return {"number": "1", "position": str(pos), "positionText": str(pos), "points": "0",
                "Driver": {"code": code}, "Constructor": {"name": "T"}, "grid": str(grid), "laps": "50",
                "status": status, "Time": {"millis": str(millis)} if millis else None}
    jr = {"MRData": {"RaceTable": {"Races": [{"Results": [
        res(1, "AAA", "Finished", 5_000_000, 1), res(2, "BBB", "Finished", 5_010_000, 2),
        res(3, "CCC", "Lapped", 5_030_000, 3), res(4, "DDD", "Retired", None, 4)]}]}}}
    r = o.convert_results(jr, pd.DataFrame(), 2025).set_index("driver")
    assert r.loc["AAA", "gap_to_winner_s"] == 0 and r.loc["BBB", "gap_to_winner_s"] == pytest.approx(10.0)
    assert np.isnan(r.loc["CCC", "gap_to_winner_s"]) and np.isnan(r.loc["DDD", "race_time_s"])


def test_lap_quality_flags_holes_and_missing_final_time():
    laps = pd.DataFrame({
        "driver": ["A"] * 4 + ["B"] * 2, "lap_number": [1, 2, 3, 4, 1, 4],
        "lap_time_seconds": [90.0, 90.0, 90.0, np.nan, 90.0, 90.0]})
    res = pd.DataFrame({"driver": ["A", "B"], "laps": [4, 4], "status": ["Finished", "Finished"]})
    q = o.lap_quality(laps, res)
    assert q["inner_missing_laps"] == 2 and q["drivers_with_gaps"] == ["B"]
    assert q["finishers_no_final_time"] == 1 and q["usable"] is False
    ok = laps[laps["driver"] == "A"].assign(lap_time_seconds=90.0)
    assert o.lap_quality(ok, res.iloc[:1])["usable"] is True


def test_pit_agreement_matches_jolpica_by_driver_and_lap():
    laps = pd.DataFrame({"driver": ["AAA", "AAA", "BBB"], "lap_number": [5, 6, 9], "pit_in_lap": [1, 0, 1]})
    jr = {"MRData": {"RaceTable": {"Races": [{"Results": [
        {"Driver": {"driverId": "a", "code": "AAA"}}, {"Driver": {"driverId": "b", "code": "BBB"}}]}]}}}
    jp = {"MRData": {"RaceTable": {"Races": [{"PitStops": [
        {"driverId": "a", "lap": "5"}, {"driverId": "b", "lap": "8"}]}]}}}
    r = o.pit_agreement(laps, jp, jr)
    assert r["matched"] == 1 and r["jolpica_stops"] == 2 and r["openf1_only"] == 1


def test_labels_cover_known_circuits():
    assert o.CIRCUIT_LABEL["Catalunya"] == "Barcelona" and o.CIRCUIT_LABEL["Madring"] == "Madrid"
    assert o.CIRCUIT_LABEL["Kuala Lumpur"] == "Malaysia"


def test_cached_get_never_refetches(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(o, "_get_json", lambda url, params=None, retries=8: calls.append(url) or [{"x": 1}])
    a = o.cached_get("k/x.json", "http://example/x", None, raw_dir=str(tmp_path))
    b = o.cached_get("k/x.json", "http://example/x", None, raw_dir=str(tmp_path))
    assert a == b == [{"x": 1}] and len(calls) == 1


def test_real_cached_session_builds_clean_laps():
    key = 11253  # Japan 2026
    if not os.path.exists(os.path.join(o.RAW_DIR, str(key), "laps.json")):
        pytest.skip("raw OpenF1 cache not present")
    meta = {"season": 2026, "round": 3, "session_key": key, "label": "Japan",
            "race_name": "Japanese Grand Prix", "circuit": "Suzuka", "date": "2026-03-29"}
    t = o.build_race(meta)
    L = t["laps"]
    assert len(L) > 1000 and L["driver"].nunique() == 22
    assert not L.duplicated(["driver", "lap_number"]).any()
    assert t["race_info"]["data_quality"]["usable"] is True
    assert t["race_info"]["data_quality"]["pit_vs_jolpica"]["match_rate"] >= 0.95
    assert t["race_info"]["compound_map"] == {"HARD": "C1", "MEDIUM": "C2", "SOFT": "C3"}
