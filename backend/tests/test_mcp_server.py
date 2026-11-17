"""T-054: MCP server tools called in-process, plus one in-memory MCP client round trip."""

import asyncio
import json

import pytest

pytest.importorskip("mcp")

from app import mcp_server as m  # noqa: E402

PLAN = [
    {"name": "1stop", "stints": [{"compound": "MEDIUM", "laps": 30}, {"compound": "HARD", "laps": 41}]},
    {"name": "2stop", "stints": [{"compound": "SOFT", "laps": 20}, {"compound": "MEDIUM", "laps": 25},
                                 {"compound": "HARD", "laps": 26}]},
]


def _has_caveats(r):
    assert isinstance(r["caveats"], list) and r["caveats"]


def test_list_tracks():
    r = m.list_tracks()
    _has_caveats(r)
    assert any(t["track_id"] == "brazil" for t in r["tracks"])


def test_optimize_strategy():
    r = m.optimize_strategy("brazil", n_sims=200, top_n=3)
    _has_caveats(r)
    assert len(r["strategies"]) == 3 and r["strategies"][0]["rank"] == 1
    assert any("1-stop" in c for c in r["caveats"])


def test_optimize_unknown_track_is_error_result():
    r = m.optimize_strategy("atlantis")
    assert "Unknown track_id" in r["error"]
    _has_caveats(r)


def test_simulate_strategy_and_validation():
    r = m.simulate_strategy(PLAN, track_id="brazil", n_simulations=300, seed=1)
    _has_caveats(r)
    assert r["reference_strategy"] == "1stop"
    assert all("lap_times" not in x for x in r["results"])
    assert m.simulate_strategy(PLAN, track_id="brazil", n_simulations=50, include_lap_times=True)["results"][0]["lap_times"]
    bad = m.simulate_strategy([{"name": "a", "stints": [{"compound": "X", "laps": 3}]}], track_id="brazil")
    assert "compound" in bad["error"]


def test_simulate_deterministic_given_seed():
    a = m.simulate_strategy(PLAN, track_id="brazil", n_simulations=200, seed=5)
    b = m.simulate_strategy(PLAN, track_id="brazil", n_simulations=200, seed=5)
    assert a["results"] == b["results"]


def test_simulate_grid_tool():
    r = m.simulate_grid("austin", n_sims=40, top_n=5, forced_sc_lap=20)
    _has_caveats(r)
    assert r["forced_sc_lap"] == 20 and len(r["drivers"]) == 5
    assert any("forced from lap 20" in c for c in r["caveats"])
    assert r["drivers"][0]["exp_points"] >= r["drivers"][-1]["exp_points"]
    assert m.simulate_grid("austin", n_sims=40, seed=2) == m.simulate_grid("austin", n_sims=40, seed=2)
    bad = m.simulate_grid("austin", n_sims=40, strategies={"NOR": [{"compound": "HARD", "laps": 3}]})
    assert "error" in bad and bad["caveats"]
    assert "error" in m.simulate_grid("nowhere")


def test_get_race_results():
    r = m.get_race_results(20)
    _has_caveats(r)
    assert r["results"][0]["position"] == 1 and len(r["results"]) >= 20
    assert "error" in m.get_race_results(99)


def test_predict_race_past_round():
    r = m.predict_race(20, top_n=3)
    _has_caveats(r)
    assert len(r["drivers"]) == 3 and r["race_done"] and 0 <= r["drivers"][0]["p_win"] <= 1
    assert "error" in m.predict_race(999)


def test_predict_race_next_default(monkeypatch):
    from app.services import prediction_service as svc

    fake = {"race": {"round": 99, "name": "X GP"}, "mode": "pre_quali", "stale": False, "generated_at": "t",
            "headline": {"winner": "AAA", "podium": ["AAA", "BBB", "CCC"]},
            "drivers": [{"code": "AAA", "name": "A", "team": "T", "p_win": 0.5, "p_podium": 0.8, "p_top10": 0.9,
                         "exp_pos": 2.0, "p_dnf": 0.1, "exp_points": 15.0, "pred_rank": 1}],
            "disclaimer": "d", "evidence": {"notes": []}}
    monkeypatch.setattr(svc, "get_next", lambda: fake)
    r = m.predict_race()
    assert r["race"]["round"] == 99 and r["drivers"][0]["code"] == "AAA"
    _has_caveats(r)


def test_query_laps_filters_and_cap():
    r = m.query_laps("Austin", driver="nor", lap_from=1, lap_to=5)
    _has_caveats(r)
    assert r["n_returned"] <= 5 and all(x["driver"] == "NOR" for x in r["rows"])
    big = m.query_laps("Austin")
    assert big["n_returned"] == m.MAX_LAP_ROWS and big["truncated"]
    assert "ambiguous" in m.query_laps("a")["error"] or "error" in m.query_laps("zzz")
    assert "unknown columns" in m.query_laps("Austin", columns=["nope"])["error"]


def test_chunking_and_bm25():
    text = "intro line\n# A\nalpha beta\n```\n# not a heading\n```\n## B\ngamma delta delta\n"
    ch = m.chunk_markdown(text, "x.md")
    assert [c["heading"] for c in ch] == ["(intro)", "A", "B"]
    assert "not a heading" in ch[1]["text"]
    idx = m.BM25Index(ch)
    assert idx.search("delta", 1)[0][1]["heading"] == "B"
    assert idx.search("zzzz") == []


def test_search_project_notes():
    r = m.search_project_notes("MCP server stdio transport", k=3)
    _has_caveats(r)
    assert 1 <= len(r["hits"]) <= 3
    h = r["hits"][0]
    assert h["path"].endswith(".md") and h["heading"] and h["snippet"]
    assert r["n_chunks_indexed"] > 50


def test_mcp_round_trip_in_memory():
    from mcp import Client

    async def go():
        async with Client(m.build_server()) as client:
            tools = {t.name for t in (await client.list_tools()).tools}
            assert {"list_tracks", "predict_race", "get_race_results", "optimize_strategy", "simulate_strategy",
                    "simulate_grid", "query_laps", "search_project_notes"} <= tools
            res = await client.call_tool("search_project_notes", {"query": "pit loss calibration", "k": 2})
            assert not res.is_error
            payload = res.structured_content or json.loads(res.content[0].text)
            assert payload["hits"] and payload["caveats"]

    asyncio.run(go())
