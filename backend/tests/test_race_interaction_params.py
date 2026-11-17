"""Schema and sign sanity for data/sim_params/race_interaction.json (T-053, written by fit_race_interaction.py)."""

import json
import math
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[2] / "data" / "sim_params" / "race_interaction.json"
TRACKS = Path(__file__).resolve().parents[2] / "data" / "tracks" / "track_params.json"


@pytest.fixture(scope="module")
def p():
    return json.loads(PATH.read_text(encoding="utf-8"))


def _sigmoid(x):
    return 1 / (1 + math.exp(-x))


def _prob(p, pace=0.0, age=0.0, gap=0.5, track=None):
    o = p["overtake"]
    eta = (o["intercept"] + o["pace_delta_coef"] * pace + o["tyre_age_delta_coef"] * age
           + o["gap_coef"] * gap + o["track_offset"].get(track, 0.0))
    return _sigmoid(eta)


def test_schema(p):
    assert p["version"] == 1 and p["fitted_on"]
    for k in ("window_s", "time_loss_s", "extra_deg_frac"):
        assert isinstance(p["dirty_air"][k], float)
    o = p["overtake"]
    assert o["model"] == "logistic"
    for k in ("intercept", "pace_delta_coef", "tyre_age_delta_coef", "gap_coef"):
        assert isinstance(o[k], float)
    assert set(p["pass_time_cost_s"]) >= {"attacker", "defender"}
    assert set(p["lap1"]["by_grid_bin"]) == {"1-3", "4-6", "7-10", "11-15", "16-22"}
    for v in p["lap1"]["by_grid_bin"].values():
        assert set(v) >= {"mean", "sd"} and v["sd"] > 0
    assert p["min_follow_gap_s"] > 0
    assert p["n_obs"]


def test_track_offsets_use_known_track_ids(p):
    ids = {t["track_id"] for t in json.loads(TRACKS.read_text(encoding="utf-8"))}
    assert set(p["overtake"]["track_offset"]) <= ids


def test_dirty_air_sane(p):
    d = p["dirty_air"]
    assert 0.5 <= d["window_s"] <= 2.0
    assert 0.0 <= d["time_loss_s"] <= 1.5
    assert 0.0 <= d["extra_deg_frac"] <= 1.0


def test_overtake_signs(p):
    assert _prob(p, pace=0.5) > _prob(p, pace=0.0) > _prob(p, pace=-0.5)  # faster attacker -> more passes
    assert _prob(p, age=8) > _prob(p, age=0) > _prob(p, age=-8)  # fresher attacker tyres -> more passes
    assert _prob(p, gap=0.3) > _prob(p, gap=0.8) > _prob(p, gap=1.4)  # bigger gap -> fewer passes
    assert 0.0 < _prob(p) < 1.0


def test_lap1_and_cost_sane(p):
    assert all(abs(v["mean"]) < 3 for v in p["lap1"]["by_grid_bin"].values())
    assert -1.0 <= p["pass_time_cost_s"]["attacker"] <= 3.0
    assert -1.0 <= p["pass_time_cost_s"]["defender"] <= 3.0
