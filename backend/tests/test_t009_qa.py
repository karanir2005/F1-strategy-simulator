"""T-009 QA: invariants on data/staging/ (skips if staging absent)."""
import glob
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
STAGING = REPO / "data" / "staging"
PROCESSED = REPO / "data" / "processed"
pytestmark = pytest.mark.skipif(not STAGING.exists(), reason="no staging data")


def _names():
    return sorted(Path(p).name[:-9] for p in glob.glob(str(STAGING / "*_laps.csv")))


def test_columns_superset_of_processed():
    base = list(pd.read_csv(PROCESSED / "China_2026_laps.csv", nrows=1).columns)
    for n in _names():
        cols = list(pd.read_csv(STAGING / f"{n}_laps.csv", nrows=1).columns)
        assert cols == base + ["season", "era", "source"], n


def test_unusable_flags():
    flags = {n: json.load(open(STAGING / f"{n}_race_info.json"))["data_quality"]["usable"] for n in _names()}
    assert flags["Miami_2025"] is False and flags["Australia_2026"] is False
    assert sum(not v for v in flags.values()) == 2


def test_no_driver_with_laps_dropped():
    for n in _names():
        res = pd.read_csv(STAGING / f"{n}_results.csv")
        laps = set(pd.read_csv(STAGING / f"{n}_laps.csv", usecols=["driver"]).driver)
        dropped = res[~res.driver.isin(laps) & (res.laps.fillna(0) > 0)]
        assert dropped.empty, (n, dropped.driver.tolist())


def test_staged_race_loads_via_scorecard_and_tire_model(tmp_path):
    sys.path.insert(0, str(REPO))
    import validate_scorecard as vs
    from app.core import tire_model as tm

    for ext in ("laps.csv", "results.csv", "race_info.json"):
        shutil.copy(STAGING / f"Barcelona_2026_{ext}", tmp_path / f"Barcelona_2026_{ext}")
    race = vs.load_race("Barcelona", processed_dir=tmp_path)
    assert race.total_laps == 66 and len(race.laps) > 1000
    laps = tm.load_all_laps(str(tmp_path / "*_laps.csv"))
    clean = tm.clean_laps(tm.assign_stint_ids(laps))
    assert len(clean) > 500
