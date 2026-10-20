"""T-003: validation scorecard v2 (repo-root script validate_scorecard.py)."""

import json
import shutil
import socket
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import validate_scorecard as vs  # noqa: E402
from app.core import jolpica_client  # noqa: E402
from app.core.race_simulator import Stint  # noqa: E402

REAL_DATA = all(
    (REPO_ROOT / "data" / "processed" / f"{gp}_2026_laps.csv").exists() for gp in ("Australia", "Austria", "Belgium")
) and (REPO_ROOT / "data" / "jolpica" / "raw").exists()
needs_data = pytest.mark.skipif(not REAL_DATA, reason="cached 2026 race data / Jolpica cache not present")


# ---------------------------------------------------------------- pure helpers


def test_spearman_perfect_reverse_and_ties():
    assert vs.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert vs.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert np.isnan(vs.spearman([1, 1, 1], [1, 2, 3]))


def test_wilson_ci_known_values_and_empty():
    lo, hi = vs.wilson_ci(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
    assert vs.wilson_ci(0, 0) == (None, None)
    lo, hi = vs.wilson_ci(10, 10)
    assert hi == pytest.approx(1.0, abs=1e-3) and lo < 1.0


def test_score_sign_convention_and_tie():
    assert vs._score(-3.0, True) == 1.0  # a faster in sim, a finished ahead
    assert vs._score(-3.0, False) == 0.0
    assert vs._score(3.0, False) == 1.0
    assert vs._score(0.0, True) == 0.5


def test_bootstrap_is_deterministic_and_brackets_mean():
    pairs = pd.DataFrame({"race": list("AAABBBCCC"), "x": [1, 1, 0, 1, 0, 0, 1, 1, 1]})
    a = vs.race_bootstrap_ci(pairs, "x")
    assert a == vs.race_bootstrap_ci(pairs, "x")
    assert a[0] <= pairs["x"].mean() <= a[1]
    assert vs.race_bootstrap_ci(pairs.iloc[0:0], "x") == (None, None)


def test_paired_race_diff_zero_when_columns_equal():
    pairs = pd.DataFrame({"race": list("AABB"), "p": [1, 0, 1, 1], "q": [1, 0, 1, 1]})
    out = vs.paired_race_diff_ci(pairs, "p", "q")
    assert out["diff"] == 0.0 and out["race_bootstrap95"] == (0.0, 0.0)


def test_stop_laps_and_strategy_key():
    st = [Stint("MEDIUM", 20), Stint("HARD", 30), Stint("SOFT", 6)]
    assert vs.strategy_key(st) == ("MEDIUM", "HARD", "SOFT")
    assert vs.stop_laps(st) == [21, 51]


# ---------------------------------------------------------------- qualifying


# jolpica_client caches the current season per round: 2026/<round>/qualifying
QUALI_FILE = jolpica_client._cache_path("2026/7/qualifying", 0).name


def _quali_json(tmp_path, race_name="Test Grand Prix"):
    body = {
        "MRData": {
            "total": "3",
            "RaceTable": {
                "Races": [
                    {
                        "raceName": race_name,
                        "QualifyingResults": [
                            {"position": "1", "Driver": {"code": "AAA"}, "Q1": "1:20.500", "Q2": "1:19.900", "Q3": "1:19.250"},
                            {"position": "2", "Driver": {"code": "BBB"}, "Q1": "1:21.000", "Q2": "1:20.400", "Q3": ""},
                            {"position": "3", "Driver": {"code": "CCC"}, "Q1": "1:22.000"},
                        ],
                    }
                ]
            }
        }
    }
    (tmp_path / QUALI_FILE).write_text(json.dumps(body), encoding="utf-8")


def test_load_jolpica_qualifying_takes_min_of_sessions(tmp_path):
    _quali_json(tmp_path)
    q = vs.load_jolpica_qualifying("Test Grand Prix", tmp_path).set_index("driver")
    assert q.at["AAA", "q_best"] == pytest.approx(79.25)
    assert q.at["BBB", "q_best"] == pytest.approx(80.4)
    assert q.at["CCC", "q_best"] == pytest.approx(82.0)
    assert vs.load_jolpica_qualifying("Other Grand Prix", tmp_path).empty


def test_quali_pace_fallbacks_teammate_then_slowest():
    quali = pd.DataFrame({"driver": ["AAA", "CCC"], "q_best": [80.0, 82.0]})
    results = pd.DataFrame(
        {
            "driver": ["AAA", "BBB", "CCC", "DDD"],
            "team": ["T1", "T1", "T2", "T3"],
        }
    )
    out = vs.quali_pace(quali, results).set_index("driver")
    assert out.at["AAA", "q_source"] == "own"
    assert (out.at["BBB", "q_best"], out.at["BBB", "q_source"]) == (80.0, "teammate")
    assert (out.at["DDD", "q_best"], out.at["DDD", "q_source"]) == (82.0, "slowest")


def test_load_jolpica_is_read_only_no_network(monkeypatch, tmp_path):
    _quali_json(tmp_path)

    def boom(*a, **k):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "socket", boom)
    assert not vs.load_jolpica_qualifying("Test Grand Prix", tmp_path).empty
    assert not (tmp_path / "x").exists()
    assert [p.name for p in tmp_path.iterdir()] == [QUALI_FILE]


# ---------------------------------------------------------------- per-driver pieces


def test_full_distance_strategy_extends_last_stint():
    d = pd.DataFrame(
        {
            "lap_number": [1, 2, 3, 4],
            "lap_time_seconds": [95.0, 90.0, 90.0, 90.0],
            "lap_start_s": [0.0, 95.0, 185.0, 275.0],
            "lap_end_s": [95.0, 185.0, 275.0, 365.0],
            "compound": ["MEDIUM"] * 4,
            "tire_age": [1.0, 2.0, 3.0, 4.0],
            "tyre_stint": [1] * 4,
            "stint_start_age": [1] * 4,
            "ff1_stint": [1] * 4,
        }
    )
    st = vs.full_distance_strategy(d, 10)
    assert [(s.compound, s.laps, s.start_age) for s in st] == [("MEDIUM", 10, 1)]


def test_actual_lap_times_falls_back_to_timing_line_diff():
    d = pd.DataFrame(
        {
            "lap_number": [1, 2],
            "lap_time_seconds": [90.0, np.nan],
            "lap_start_s": [0.0, 90.0],
            "lap_end_s": [90.0, 250.0],
        }
    )
    assert vs.actual_lap_times(d).tolist() == [90.0, 160.0]


def test_summarise_stint_errors_relative_removes_driver_level():
    df = pd.DataFrame(
        {
            "race": ["R"] * 4,
            "driver": ["A", "A", "B", "B"],
            "err": [2.0, 3.0, -1.0, -1.0],
            "n_laps": [10] * 4,
        }
    )
    s = vs.summarise_stint_errors(df)
    assert s["bias_s_per_lap"] == pytest.approx(0.75)
    assert s["mae_s_per_lap"] == pytest.approx(1.75)
    assert s["relative_mae_s_per_lap"] == pytest.approx(0.25)  # |-.5|,|.5|,0,0
    assert vs.summarise_stint_errors(df.iloc[0:0]) == {"n_stints": 0}


# ---------------------------------------------------------------- end to end on real cached data


def _copy_races(tmp_path, gps):
    src = REPO_ROOT / "data" / "processed"
    for gp in gps:
        for suffix in ("laps.csv", "results.csv", "race_info.json"):
            shutil.copy(src / f"{gp}_2026_{suffix}", tmp_path / f"{gp}_2026_{suffix}")
    return tmp_path


@pytest.fixture(scope="module")
def mini_card(tmp_path_factory):
    d = _copy_races(tmp_path_factory.mktemp("proc"), ["Australia", "Austria", "Belgium"])
    return vs.run_scorecard(["Australia", "Austria", "Belgium"], d)


@needs_data
def test_scorecard_structure_and_json_roundtrip(mini_card, tmp_path):
    for key in ("meta", "headline", "base_laptime", "leakage_cost", "per_stint", "finishing_order", "strategy_ranking", "race_totals"):
        assert key in mini_card
    assert mini_card["meta"]["primary_base_method"] == "quali_ratio"
    assert set(mini_card["per_stint"]["summary"]) == set(vs.BASE_METHODS)
    out = tmp_path / "s" / "latest.json"
    vs.write_scorecard(mini_card, out)
    back = json.loads(out.read_text(encoding="utf-8"))
    assert back["headline"]["per_stint_mae_s_per_lap"] == mini_card["headline"]["per_stint_mae_s_per_lap"]


@needs_data
def test_scorecard_is_deterministic(mini_card, tmp_path_factory):
    d = _copy_races(tmp_path_factory.mktemp("proc2"), ["Australia", "Austria", "Belgium"])
    again = vs.run_scorecard(["Australia", "Austria", "Belgium"], d)
    for card in (mini_card, again):
        card["meta"].pop("generated_at")
    assert json.dumps(mini_card, default=vs._json_default, sort_keys=True) == json.dumps(
        again, default=vs._json_default, sort_keys=True
    )


@needs_data
def test_neutralised_laps_dominate_incl_variant(mini_card):
    s = mini_card["per_stint"]["summary"]["quali_ratio"]
    assert s["incl_neutralised"]["mae_s_per_lap"] >= s["excl_neutralised"]["mae_s_per_lap"]
    # the oracle floor must not be worse than the non-leaky base on the headline variant
    o = mini_card["per_stint"]["summary"]["oracle"]["excl_neutralised"]["mae_s_per_lap"]
    assert o <= s["excl_neutralised"]["mae_s_per_lap"]


@needs_data
def test_loro_and_quali_ratio_never_see_the_scored_race(tmp_path_factory):
    """
    Corrupt Austria's lap times by +15 s plus 0.3 s/lap (a constant alone would be absorbed by the fixed effects). Austria's own LORO tyre model and
    quali_ratio must be bit-identical (they are fitted on the other races);
    Australia's model, which trains on Austria, must change.
    """
    gps = ["Australia", "Austria", "Belgium"]
    clean_dir = _copy_races(tmp_path_factory.mktemp("clean"), gps)
    dirty_dir = _copy_races(tmp_path_factory.mktemp("dirty"), gps)
    p = dirty_dir / "Austria_2026_laps.csv"
    laps = pd.read_csv(p)
    laps["lap_time_seconds"] = laps["lap_time_seconds"] + 15.0 + 0.3 * laps["lap_number"]
    laps.to_csv(p, index=False)

    a = vs.run_scorecard(gps, clean_dir)
    b = vs.run_scorecard(gps, dirty_dir)

    assert a["base_laptime"]["Austria"]["quali_ratio"] == b["base_laptime"]["Austria"]["quali_ratio"]
    assert a["base_laptime"]["Austria"]["loro_model"] == b["base_laptime"]["Austria"]["loro_model"]
    assert a["base_laptime"]["Australia"]["loro_model"] != b["base_laptime"]["Australia"]["loro_model"]
    assert a["base_laptime"]["Australia"]["quali_ratio"] != b["base_laptime"]["Australia"]["quali_ratio"]
    # Austria's primary-method grid/sim finishing order is built from quali + the LORO model only
    assert a["finishing_order"]["per_race"]["Austria"]["sim_rho_quali_ratio"] == b["finishing_order"]["per_race"]["Austria"]["sim_rho_quali_ratio"]
    # the perturbation does bite: the non-leaky prediction is now ~15 s/lap off the (corrupted) actuals
    key = "by_race_quali_ratio_excl_neutralised"
    assert b["per_stint"][key]["Austria"]["bias_s_per_lap"] < a["per_stint"][key]["Austria"]["bias_s_per_lap"] - 10


@needs_data
def test_strategy_pairs_are_similar_pace_and_different_strategies(mini_card):
    pairs = pd.DataFrame(mini_card["strategy_ranking"]["pairs"])
    if pairs.empty:
        pytest.skip("no pairs in the 3-race subset")
    assert (pairs["quali_gap_pct"].abs() <= 100 * vs.SIMILAR_PACE_PCT + 1e-6).all()
    assert (pairs["strategy_a"] != pairs["strategy_b"]).all()
    assert set(pairs["kind"]) <= {"different_sequence", "timing_only"}
    assert pairs["sim_correct"].isin([0.0, 0.5, 1.0]).all()
    prim = mini_card["strategy_ranking"]["primary"]
    assert prim["n_pairs"] == int((pairs["kind"] == "different_sequence").sum())


@needs_data
def test_committed_latest_json_matches_schema():
    latest = REPO_ROOT / "data" / "scorecard" / "latest.json"
    if not latest.exists():
        pytest.skip("run validate_scorecard.py first")
    card = json.loads(latest.read_text(encoding="utf-8"))
    assert card["meta"]["scorecard_version"] == vs.SCORECARD_VERSION
    assert len(card["meta"]["races"]) == 16  # T-015: 10 + 6 promoted
    assert card["strategy_ranking"]["primary"]["n_pairs"] > 0
