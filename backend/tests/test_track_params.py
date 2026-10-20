"""Schema + sanity checks for data/tracks/track_params.json (T-016)."""

import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "data" / "tracks" / "track_params.json"
SCHEDULE = ROOT / "data" / "jolpica" / "derived" / "schedule.csv"

KEYS = {
    "track_id", "name", "circuit", "total_laps", "pit_loss_s", "pit_loss_noise_std_s",
    "base_laptime_s", "compounds", "max_stint_laps", "typical_stint_laps", "source_season", "source_races",
    "on_2026_calendar", "already_raced_2026", "notes",
}
DRY = {"SOFT", "MEDIUM", "HARD"}


@pytest.fixture(scope="module")
def tracks():
    if not PATH.exists():
        pytest.skip("run build_track_params.py first")
    return json.loads(PATH.read_text(encoding="utf-8"))


def test_schema(tracks):
    assert tracks
    for t in tracks:
        assert set(t) == KEYS, t["track_id"]
        assert isinstance(t["track_id"], str) and t["track_id"]
        assert isinstance(t["total_laps"], int)
        for k in ("pit_loss_s", "pit_loss_noise_std_s", "base_laptime_s"):
            assert isinstance(t[k], (int, float)) and t[k] == t[k], (t["track_id"], k)
        assert t["source_season"] in (2025, 2026)
        assert t["source_races"] and all(isinstance(r, str) for r in t["source_races"])
        assert t["on_2026_calendar"] is True
        assert isinstance(t["already_raced_2026"], bool)
        assert isinstance(t["notes"], str)
        if t["compounds"] is not None:
            assert set(t["compounds"]) == {"HARD", "MEDIUM", "SOFT"}
        assert t["max_stint_laps"] and set(t["max_stint_laps"]) <= DRY


def test_unique_ids(tracks):
    ids = [t["track_id"] for t in tracks]
    assert len(ids) == len(set(ids))


def test_every_2026_calendar_circuit_present(tracks):
    sched = pd.read_csv(SCHEDULE)
    n = int((sched["season"] == 2026).sum())
    assert len(tracks) == n


def test_sane_ranges(tracks):
    for t in tracks:
        tid = t["track_id"]
        assert 40 <= t["total_laps"] <= 80, tid
        assert 15 <= t["pit_loss_s"] <= 35, tid
        assert 0 < t["pit_loss_noise_std_s"] <= 10, tid
        assert 60 <= t["base_laptime_s"] <= 125, tid
        for c, n in t["max_stint_laps"].items():
            assert 5 <= n <= t["total_laps"], (tid, c)


def test_2025_fallbacks_flagged(tracks):
    for t in tracks:
        if t["source_season"] == 2025:
            assert not t["already_raced_2026"], t["track_id"]
            assert "2025" in t["notes"], t["track_id"]
            assert all(r.endswith("_2025") for r in t["source_races"])


def test_austin_matches_contract_example(tracks):
    a = next(t for t in tracks if t["track_id"] == "austin")
    assert a["total_laps"] == 56 and a["source_season"] == 2025
    assert a["compounds"] == {"HARD": "C1", "MEDIUM": "C3", "SOFT": "C4"}
