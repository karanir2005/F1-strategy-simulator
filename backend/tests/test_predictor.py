"""
Tests for the race-outcome predictor (T-011) and its Jolpica client.

Everything except the last (real-data smoke) test runs on a small
synthetic season, so no network and no data/jolpica/ files are needed.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.core import jolpica_client as J
from app.core import predictor as P

REPO = Path(__file__).resolve().parents[2]
DERIVED = REPO / "data" / "jolpica" / "derived"


# ------------------------------------------------------------------ fixtures


def _synthetic_history(n_rounds=8, seed=0):
    """
    5 teams x 2 drivers. True strength decreases with team index; driver
    'a' of each team is a bit stronger than 'b'. Team t4 retires a lot.
    """
    rng = np.random.default_rng(seed)
    teams = [f"t{i}" for i in range(5)]
    drivers = [(f"{t}a", t) for t in teams] + [(f"{t}b", t) for t in teams]
    strength = {d: -0.8 * int(t[1]) + (0.3 if d.endswith("a") else 0.0) for d, t in drivers}
    res_rows, q_rows = [], []
    for season in (2025, 2026):
        for rnd in range(1, n_rounds + 1):
            q_perf = {d: strength[d] + rng.gumbel() for d, _ in drivers}
            q_order = sorted(q_perf, key=q_perf.get, reverse=True)
            grid = {d: i + 1 for i, d in enumerate(q_order)}
            r_perf = {d: strength[d] - 0.3 * np.log(grid[d]) + rng.gumbel() for d, _ in drivers}
            dnf = {d: rng.random() < (0.5 if t == "t4" else 0.1) for d, t in drivers}
            fin = sorted([d for d, _ in drivers if not dnf[d]], key=r_perf.get, reverse=True)
            out = sorted([d for d, _ in drivers if dnf[d]], key=r_perf.get, reverse=True)
            for pos, d in enumerate(fin + out, start=1):
                t = d[:2]
                classified = d in fin
                res_rows.append(
                    dict(season=season, round=rnd, race_name=f"R{rnd}", circuit_id=f"c{rnd}",
                         date=f"{season}-01-{rnd:02d}", session="race", driver_id=d,
                         code=d.upper(), team_id=t, team=t.upper(), position=pos,
                         position_text=str(pos) if classified else "R", classified=classified,
                         grid=grid[d], status="Finished" if classified else "Retired",
                         points=float(max(0, 11 - pos)) if classified else 0.0, laps=50)
                )
            for pos, d in enumerate(q_order, start=1):
                q_rows.append(dict(season=season, round=rnd, circuit_id=f"c{rnd}", driver_id=d,
                                   code=d.upper(), team_id=d[:2], team=d[:2].upper(),
                                   quali_position=pos))
    return pd.DataFrame(res_rows), pd.DataFrame(q_rows)


@pytest.fixture
def hist():
    r, q = _synthetic_history()
    return P.prepare_history(r, q)


FAST = P.PredictorConfig(n_sims=4000, seed=3, topk_level2=5, topk_quali=5)


# --------------------------------------------------------- Plackett-Luce core


@pytest.mark.parametrize("topk", [None, 3])
def test_pl_gradient_matches_finite_differences(topk):
    rng = np.random.default_rng(1)
    u = rng.normal(size=8)
    ll, g = P.pl_loglik_grad(u, topk)
    eps = 1e-6
    num = np.array([(P.pl_loglik_grad(u + eps * e, topk)[0] - P.pl_loglik_grad(u - eps * e, topk)[0]) / (2 * eps)
                    for e in np.eye(len(u))])
    assert np.allclose(g, num, atol=1e-6)
    assert ll < 0


def test_pl_two_items_is_logistic():
    ll, _ = P.pl_loglik_grad(np.array([1.0, 0.0]))
    assert ll == pytest.approx(np.log(1 / (1 + np.exp(-1.0))))


def test_fit_pl_recovers_strength_order():
    # item 0 always beats 1 beats 2 -> fitted utilities ordered 0 > 1 > 2
    X = np.eye(3)
    x = P.fit_pl([X] * 10, [1.0] * 10, [0.1, 0.1, 0.1])
    assert x[0] > x[1] > x[2]


# -------------------------------------------------------------- simulation


def test_simulation_is_coherent_and_seeded():
    model = P.OutcomeModel(b_race=1.0, b_grid=1.0, a_quali=1.0)
    th = np.linspace(1, -1, 10)
    p_dnf = np.full(10, 0.2)
    for mode, grid in (("grid", np.arange(1, 11, dtype=float)), ("prequali", None)):
        a = P.simulate(th, th, grid, p_dnf, model, mode, 5000, 7)
        b = P.simulate(th, th, grid, p_dnf, model, mode, 5000, 7)
        assert np.array_equal(a["p_win"], b["p_win"])
        assert a["p_win"].sum() == pytest.approx(1.0)
        assert a["p_podium"].sum() == pytest.approx(3.0)
        assert a["p_top10"].sum() == pytest.approx(10.0)
        assert np.all(a["p_win"] <= a["p_podium"]) and np.all(a["p_podium"] <= a["p_top10"])
        assert a["p_win"][0] > a["p_win"][-1]
        # a driver with a 20% DNF chance can't be on the podium more than 80% of the time
        assert a["p_podium"].max() <= 0.8 + 0.02


def test_dnf_component_caps_win_probability():
    model = P.OutcomeModel(b_race=10.0, b_grid=0.0, a_quali=0.0)
    th = np.r_[5.0, np.zeros(9)]
    out = P.simulate(th, th, np.arange(1, 11, dtype=float), np.r_[0.3, np.zeros(9)], model, "grid", 20000, 1)
    assert out["p_win"][0] == pytest.approx(0.7, abs=0.02)


# ----------------------------------------------------------- DNF shrinkage


def test_dnf_probabilities_shrink_and_rank_teams(hist):
    ent = P.entrants_for(hist, 2026, 8)
    p = P.dnf_probabilities(hist, 2026, 8, ent, FAST)
    assert np.all((p > 0) & (p < 1))
    by_team = pd.Series(p, index=ent["team_id"]).groupby(level=0).mean()
    assert by_team.idxmax() == "t4"
    # no history at round 1 -> everyone gets the prior
    p1 = P.dnf_probabilities(hist, 2026, 1, P.entrants_for(hist, 2026, 1), FAST)
    assert np.allclose(p1, FAST.dnf_prior_rate)
    # unknown driver in a known team falls back to the team rate
    new = pd.DataFrame({"driver_id": ["rookie"], "team_id": ["t4"]})
    assert P.dnf_probabilities(hist, 2026, 8, new, FAST)[0] > FAST.dnf_prior_rate


# --------------------------------------------------------------- prediction


@pytest.mark.parametrize("mode", ["grid", "prequali"])
def test_prediction_table_is_coherent(hist, mode):
    t = P.predict_race(hist, 2026, 8, mode, FAST)
    assert len(t) == 10
    assert t["p_win"].sum() == pytest.approx(1.0)
    assert t["p_podium"].sum() == pytest.approx(3.0)
    assert t["p_top10"].sum() == pytest.approx(10.0)
    # strongest synthetic team should lead, weakest should trail
    assert t.iloc[0]["team_id"] in ("t0", "t1")
    assert t.groupby("team_id")["p_win"].sum().idxmin() in ("t3", "t4")


def test_grid_mode_requires_a_grid(hist):
    ent = P.entrants_for(hist, 2026, 8)[["driver_id", "code", "team_id", "team"]]
    with pytest.raises(ValueError):
        P.predict_race(hist, 2026, 99, "grid", FAST, entry_list=ent, grid=np.full(len(ent), np.nan))


def test_grid_override_changes_grid_mode_only(hist):
    ent = P.entrants_for(hist, 2026, 8)
    rev = np.arange(len(ent), 0, -1)
    base = P.predict_race(hist, 2026, 8, "grid", FAST).set_index("driver_id")
    flipped = P.predict_race(hist, 2026, 8, "grid", FAST, entry_list=ent, grid=rev).set_index("driver_id")
    assert not np.allclose(base.loc[ent["driver_id"], "p_win"], flipped.loc[ent["driver_id"], "p_win"])


# ------------------------------------------------------------------ leakage


def _scramble_target_and_future(hist, season, rnd, seed=99):
    """Reverse/garble everything about round >= rnd except the target's
    entry list and grid: positions, classification, points, status, and
    every later round's quali and results."""
    rng = np.random.default_rng(seed)
    res = hist["results"].copy()
    m = (res["season"] == season) & (res["round"] >= rnd)
    res.loc[m, "position"] = rng.permutation(res.loc[m, "position"].to_numpy())
    res.loc[m, "classified"] = rng.random(m.sum()) < 0.5
    res.loc[m, "points"] = rng.integers(0, 26, m.sum()).astype(float)
    res.loc[m, "status"] = "Scrambled"
    later = (res["season"] == season) & (res["round"] > rnd)
    res.loc[later, "grid"] = rng.permutation(res.loc[later, "grid"].to_numpy())
    q = hist["qualifying"].copy()
    mq = (q["season"] == season) & (q["round"] >= rnd)
    q.loc[mq, "quali_position"] = rng.permutation(q.loc[mq, "quali_position"].to_numpy())
    return {"results": res, "qualifying": q}


@pytest.mark.parametrize("mode", ["grid", "prequali"])
@pytest.mark.parametrize("w2025,kappa", [(0.0, 0.0), (0.5, 0.5)])
def test_no_leakage_from_target_or_future_races(hist, mode, w2025, kappa):
    cfg = P.PredictorConfig(n_sims=3000, seed=5, w2025_level2=w2025, driver_prior_2025=kappa)
    a = P.predict_race(hist, 2026, 6, mode, cfg).set_index("driver_id").sort_index()
    b = P.predict_race(_scramble_target_and_future(hist, 2026, 6), 2026, 6, mode, cfg)
    b = b.set_index("driver_id").sort_index()
    for col in ("p_win", "p_podium", "p_top10", "p_dnf", "theta_race", "theta_quali"):
        assert np.allclose(a[col], b[col]), col


def test_prequali_ignores_target_grid(hist):
    a = P.predict_race(hist, 2026, 6, "prequali", FAST).set_index("driver_id").sort_index()
    h2 = {"results": hist["results"].copy(), "qualifying": hist["qualifying"].copy()}
    m = (h2["results"]["season"] == 2026) & (h2["results"]["round"] == 6)
    h2["results"].loc[m, "grid"] = h2["results"].loc[m, "grid"].to_numpy()[::-1]
    b = P.predict_race(h2, 2026, 6, "prequali", FAST).set_index("driver_id").sort_index()
    assert np.allclose(a["p_win"], b["p_win"])


@pytest.mark.parametrize("kind", ["grid", "champ", "form"])
def test_baselines_no_leakage_and_coherent(hist, kind):
    a = P.RankBaseline(hist, kind, table_seasons=(2025,)).predict(2026, 6).set_index("driver_id").sort_index()
    h2 = _scramble_target_and_future(hist, 2026, 6)
    b = P.RankBaseline(h2, kind, table_seasons=(2025,)).predict(2026, 6).set_index("driver_id").sort_index()
    assert np.allclose(a["p_win"], b["p_win"]) and np.allclose(a["p_podium"], b["p_podium"])
    assert a["p_win"].sum() == pytest.approx(1.0, abs=1e-3)
    assert a["p_podium"].sum() == pytest.approx(3.0, abs=1e-2)


def test_walk_forward_and_scores(hist):
    wf = P.walk_forward(hist, 2026, range(2, 9), "model_grid", FAST)
    assert set(wf["round"]) == set(range(2, 9))
    assert (wf.groupby("round")["y_win"].sum() == 1).all()
    s = P.score(wf)
    u = P.score(P.walk_forward(hist, 2026, range(2, 9), "uniform", FAST))
    # 7 synthetic races x 10 cars is too few for a stable win log-loss
    # comparison; podium (3 events/race) is the robust check here.
    assert s["podium_logloss"] < u["podium_logloss"]
    cal = P.calibration_table(wf, "podium")
    assert cal["n"].sum() == len(wf)


# ------------------------------------------------------------ Jolpica client


def _page(total, rows, offset=0, leaf="Results"):
    return {"MRData": {"total": str(total), "offset": str(offset), "RaceTable": {"Races": [
        {"season": "2026", "round": "18", "raceName": "United States Grand Prix",
         "Circuit": {"circuitId": "americas"}, "date": "2026-10-25", leaf: rows}] if rows else []}}}


def test_client_does_not_cache_empty_responses(tmp_path, monkeypatch):
    monkeypatch.setattr(J, "RAW_DIR", tmp_path)
    calls = []

    def fake(path, offset):
        calls.append((path, offset))
        return _page(0, [])

    monkeypatch.setattr(J, "_http_get", fake)
    assert J.qualifying(2026, 18).empty
    assert J.qualifying(2026, 18).empty
    assert len(calls) == 2  # refetched: empty pages are never cached
    assert not list(tmp_path.iterdir())
    # offline mode never touches the network
    assert J.qualifying(2026, 18, offline=True).empty
    assert len(calls) == 2


def test_client_caches_and_merges_pages(tmp_path, monkeypatch):
    monkeypatch.setattr(J, "RAW_DIR", tmp_path)
    monkeypatch.setattr(J, "PAGE_LIMIT", 2)

    def row(code, pos):
        return {"position": str(pos), "Driver": {"driverId": code.lower(), "code": code},
                "Constructor": {"constructorId": "x", "name": "X"}, "Q1": "1:30.000"}

    pages = {0: _page(3, [row("AAA", 1), row("BBB", 2)], 0, "QualifyingResults"),
             2: _page(3, [row("CCC", 3)], 2, "QualifyingResults")}
    calls = []

    def fake(path, offset):
        calls.append(offset)
        return pages[offset]

    monkeypatch.setattr(J, "_http_get", fake)
    q = J.qualifying(2026, 18)
    assert list(q["code"]) == ["AAA", "BBB", "CCC"]
    assert list(q["quali_position"]) == [1, 2, 3]
    q2 = J.qualifying(2026, 18)
    assert calls == [0, 2]  # second call fully served from disk
    assert q2.equals(q)
    cached = json.loads(next(tmp_path.iterdir()).read_text())
    assert "MRData" in cached


# ------------------------------------------------------- real data (smoke)


@pytest.mark.skipif(not (DERIVED / "results.csv").exists(), reason="run predict_race.py once to pull Jolpica data")
def test_real_data_round17_sanity():
    r = pd.read_csv(DERIVED / "results.csv")
    q = pd.read_csv(DERIVED / "qualifying.csv")
    h = P.prepare_history(r, q)
    t = P.predict_race(h, 2026, 17, "grid", P.PredictorConfig(n_sims=5000))
    assert len(t) == 22
    assert t["p_win"].sum() == pytest.approx(1.0)
    pole = t[t["grid"] == 1].iloc[0]
    # T-010 sanity range: pole-sitter 35-60% (allow some slack), never > 75%
    assert 0.25 < pole["p_win"] < 0.75
    assert (t["p_dnf"] > 0.03).all()


# ------------------------------------------------ blend, FP feature, OpenF1


def test_blend_keeps_sums_and_exposes_components(hist):
    cfg = P.PredictorConfig(n_sims=3000, seed=2, blend_baseline=0.25)
    t = P.predict_race(hist, 2026, 8, "grid", cfg)
    for ev, tot in (("win", 1), ("podium", 3)):
        assert t[f"p_{ev}_model"].sum() == pytest.approx(tot)
        assert t[f"p_{ev}_baseline"].sum() == pytest.approx(tot, abs=1e-6)
        assert np.allclose(t[f"p_{ev}"], 0.75 * t[f"p_{ev}_model"] + 0.25 * t[f"p_{ev}_baseline"])


def test_baseline_for_future_race_uses_given_entrants(hist):
    ent = P.entrants_for(hist, 2026, 8)[["driver_id"]].assign(grid=range(10, 0, -1))
    b = P.RankBaseline(hist, "grid", table_seasons=(2025,)).predict(2026, 99, entrants=ent)
    m = b.merge(ent, on="driver_id")
    assert (m["rank"] == m["grid"]).all()  # ranked by the supplied grid
    assert b["p_win"].sum() == pytest.approx(1.0)


def test_fp_feature_is_team_level_and_neutral_without_data(hist):
    ent = P.entrants_for(hist, 2026, 8)
    assert np.all(P._fp_feature(hist, 2026, 8, ent) == 0)
    fp = pd.DataFrame({"season": 2026, "round": 8, "code": ["T0A", "T0B", "T1A"], "fp_gap": [0.0, 0.4, 1.0]})
    h2 = P.prepare_history(hist["results"], hist["qualifying"], fp=fp)
    f = pd.Series(P._fp_feature(h2, 2026, 8, ent), index=ent["code"])
    assert f["T0A"] == f["T0B"] == 0.0  # team t0 mean gap 0.2 is the best -> 0
    assert f["T1A"] == f["T1B"] == pytest.approx(-0.8)
    assert f["T3A"] == pytest.approx(-0.4)  # no long run -> field median (0.6) - 0.2


def test_long_run_stints_filters_out_laps_and_short_runs():
    from app.core import openf1_practice as F

    laps = pd.DataFrame({
        "driver_number": [1] * 10 + [2] * 4,
        "lap_number": list(range(1, 11)) + list(range(1, 5)),
        "lap_duration": [None, 100, 95.0, 95.2, 95.1, 95.3, 120.0, 95.4, 95.2, 95.0, None, 96, 96, 96],
        "is_pit_out_lap": [True] + [False] * 9 + [True, False, False, False],
    })
    stints = pd.DataFrame({"driver_number": [1, 2], "lap_start": [1, 1], "lap_end": [10, 4],
                           "compound": ["MEDIUM", "MEDIUM"]})
    lr = F.long_run_stints(laps, stints)
    assert list(lr["driver_number"]) == [1]  # driver 2's 3-lap run is too short
    row = lr.iloc[0]
    assert row["n"] == 7  # lap 1 (out/first), lap 2 (100 s, slow) and lap 7 (120 s) dropped
    assert row["pace"] == pytest.approx(95.2)


def test_2025_coef_prior_never_used_when_predicting_2025(hist):
    cfg = P.PredictorConfig(w2025_level2=1.0)
    races = P._training_races(hist, 2025, 4, cfg)
    assert races == [(2025, 1, 1.0), (2025, 2, 1.0), (2025, 3, 1.0)]


# --------------------------------------------- T-014 track grid stickiness


@pytest.fixture
def hist_rep():
    """Same synthetic seasons but circuits repeat (c0,c1,c2,...), so a
    circuit has history before the target; plus a 2024 'track_results'."""
    r, q = _synthetic_history()
    r["circuit_id"] = "c" + ((r["round"] - 1) % 3).astype(str)
    old = r[r["season"] == 2025].copy()
    old["season"] = 2024
    return P.prepare_history(r, q, track_results=old)


def _stick_cfg(**kw):
    return P.PredictorConfig(n_sims=3000, seed=5, stick_gamma=4.0, stick_k=2.0, **kw)


def test_stickiness_off_is_identical_to_baseline_model(hist_rep):
    a = P.predict_race(hist_rep, 2026, 6, "grid", P.PredictorConfig(n_sims=3000, seed=5))
    b = P.predict_race(hist_rep, 2026, 6, "grid", P.PredictorConfig(n_sims=3000, seed=5, stick_gamma=0.0, stick_k=9.0))
    pd.testing.assert_frame_equal(a, b)


def test_stickiness_shrinks_to_global_and_unknown_circuit_is_neutral(hist_rep):
    cfg = P.PredictorConfig(stick_gamma=4.0)
    st = P.track_stickiness(hist_rep, 2026, 6, cfg)
    assert st["n_cur"] == 1 and st["n_old"] == 4  # c2: 2026 R3; 2025 and 2024 R3,R6
    lo = [(P.track_stickiness(hist_rep, 2026, 6, P.PredictorConfig(stick_gamma=4.0, stick_k=k))["s_circuit"]) for k in (0.5, 5, 500)]
    # larger k pulls the circuit value monotonically toward the global value
    g = st["s_global"]
    assert abs(lo[2] - g) <= abs(lo[1] - g) + 1e-12 <= abs(lo[0] - g) + 1e-12
    # brand-new circuit: neutral
    h2 = {**hist_rep, "schedule": pd.DataFrame([dict(season=2026, round=20, circuit_id="madring")])}
    h2.pop("_rho", None)
    st2 = P.track_stickiness(h2, 2026, 20, cfg)
    assert st2["mult"] == 1.0 and st2["n_cur"] == 0 and st2["n_old"] == 0
    # first-ever race: nothing before it
    assert P.track_stickiness(hist_rep, 2024, 1, cfg)["mult"] == 1.0


def test_stickiness_ordering_and_sign():
    # circuit A: finish == grid (rho 1), circuit B: finish reverses grid (rho -1)
    rows = []
    for rnd, cid, rev in [(1, "A", False), (2, "B", True), (3, "A", False), (4, "B", True), (5, "A", False)]:
        for gpos in range(1, 11):
            rows.append(dict(season=2026, round=rnd, circuit_id=cid, driver_id=f"d{gpos}", grid=gpos,
                             position=(11 - gpos) if rev else gpos, classified=True))
    h = {"results": pd.DataFrame(rows)}
    cfg = P.PredictorConfig(stick_gamma=2.0, stick_k=1.0)
    h["schedule"] = pd.DataFrame([dict(season=2026, round=6, circuit_id="A"), dict(season=2026, round=7, circuit_id="B")])
    assert P.track_stickiness(h, 2026, 6, cfg)["mult"] > 1.0
    assert P.track_stickiness(h, 2026, 7, cfg)["mult"] < 1.0


@pytest.mark.parametrize("mode", ["grid", "prequali"])
def test_stickiness_no_leakage_from_target_or_future(hist_rep, mode):
    cfg = _stick_cfg()
    base = P.track_stickiness(hist_rep, 2026, 6, cfg)
    scr = _scramble_target_and_future(hist_rep, 2026, 6)
    # the scrambled copy must be rebuilt from raw tables (no stale caches)
    scr = P.prepare_history(scr["results"], scr["qualifying"], track_results=hist_rep["track_results"])
    st = P.track_stickiness(scr, 2026, 6, cfg)
    for k in ("mult", "s_circuit", "s_global", "n_cur", "n_old"):
        assert st[k] == pytest.approx(base[k]), k
    a = P.predict_race(hist_rep, 2026, 6, mode, cfg).set_index("driver_id").sort_index()
    b = P.predict_race(scr, 2026, 6, mode, cfg).set_index("driver_id").sort_index()
    for col in ("p_win", "p_podium", "p_top10"):
        assert np.allclose(a[col], b[col]), col


def test_stickiness_uses_only_strictly_earlier_races(hist_rep):
    cfg = P.PredictorConfig(stick_gamma=4.0)
    t = P._rho_table(hist_rep)
    for rnd in (3, 6):
        st = P.track_stickiness(hist_rep, 2026, rnd, cfg)
        sub = t[(t["season"] < 2026) | ((t["season"] == 2026) & (t["round"] < rnd))]
        w = np.where(sub["season"] == 2026, 1.0, cfg.stick_w_old)
        assert st["s_global"] == pytest.approx((w * sub["rho"]).sum() / w.sum())


@pytest.mark.parametrize("mode", ["grid", "prequali"])
def test_stickiness_keeps_sums_coherent(hist_rep, mode):
    t = P.predict_race(hist_rep, 2026, 7, mode, _stick_cfg())
    assert t["p_win"].sum() == pytest.approx(1.0)
    assert t["p_podium"].sum() == pytest.approx(3.0)
    assert t["p_top10"].sum() == pytest.approx(10.0)
    assert "stickiness" in t.attrs


def test_by_race_table_columns(hist_rep):
    t = P.by_race_table(hist_rep, 2026, range(2, 5), P.PredictorConfig(n_sims=1500, seed=1))
    assert len(t) == 3 and (t["p_actual_winner"] > 0).all()
    assert np.allclose(t["winner_nll"], -np.log(t["p_actual_winner"]))
    assert t["spearman_grid_result"].between(-1, 1).all()
