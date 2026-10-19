"""T-005 QA: independent checks of the committed scorecard (offline, reads data/scorecard/latest.json + data/processed)."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
LATEST = ROOT / "data" / "scorecard" / "latest.json"
PROC = ROOT / "data" / "processed"

pytestmark = pytest.mark.skipif(not LATEST.exists(), reason="no scorecard")


@pytest.fixture(scope="module")
def card():
    return json.loads(LATEST.read_text(encoding="utf-8"))


def test_grid_rho_matches_independent_spearman(card):
    for gp, f in card["finishing_order"]["per_race"].items():
        r = pd.read_csv(PROC / f"{gp}_2026_results.csv")
        r = r[r["status"].isin(["Finished", "Lapped"])]
        if gp in ("Canada",):
            r = r[~r["driver"].isin(card["meta"]["skipped_drivers"][gp])]
        if gp == "Hungary":
            r = r[~r["driver"].isin(card["meta"]["skipped_drivers"][gp])]
        g = r["grid_position"].where(r["grid_position"] > 0, 99)
        rho = g.rank().corr(r["position"].rank())
        assert abs(rho - f["grid_rho"]) < 1e-3, gp
        assert f["n_drivers"] == len(r), gp


def test_pair_labels_match_results(card):
    res = {gp: pd.read_csv(PROC / f"{gp}_2026_results.csv").set_index("driver") for gp in card["meta"]["races"]}
    pairs = card["strategy_ranking"]["pairs"]
    assert len(pairs) == 190
    for p in pairs:
        r = res[p["race"]]
        assert r.at[p["a"], "status"] in ("Finished", "Lapped") and r.at[p["b"], "status"] in ("Finished", "Lapped")
        assert p["actual_a_ahead"] == bool(r.at[p["a"], "position"] < r.at[p["b"], "position"])
        assert abs(p["quali_gap_pct"]) <= 0.5 + 1e-9
        # sim_correct consistent with sim_diff sign
        assert p["sim_correct"] == float((p["sim_diff_s"] < 0) == p["actual_a_ahead"]) or p["sim_diff_s"] == 0


def test_skipped_and_fallback_drivers(card):
    sk = {gp: sorted(v) for gp, v in card["meta"]["skipped_drivers"].items() if v}
    assert set(sk) == {"Canada", "Hungary"}
    assert sk["Hungary"] == ["PER"] and len(sk["Canada"]) == 7
    fb = {gp: b["quali_fallback_drivers"] for gp, b in card["base_laptime"].items() if b["quali_fallback_drivers"]}
    assert fb == {"Australia": ["VER", "SAI", "STR"], "Miami": ["HAD"]}


def test_ratio_stable_and_headline_sane(card):
    ratios = [b["quali_ratio"] for b in card["base_laptime"].values()]
    assert 1.03 < min(ratios) and max(ratios) < 1.045
    h = card["headline"]
    assert h["per_stint_mae_s_per_lap"] > card["leakage_cost"]["per_stint_mae_excl_neutralised"]["oracle"]
    assert 0 <= h["strategy_ranking_accuracy"] <= 1
