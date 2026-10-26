"""
Race-outcome predictor (T-011, 2026-10-19): P(win), P(podium), P(top 10)
per driver for one race, from information available before that race.

Two modes:
- "grid"     : the starting grid of the target race is known (Saturday
               evening onwards). Uses the post-penalty grid.
- "prequali" : before qualifying. The grid is unknown, so it is simulated
               from each driver's recent qualifying form (an explicit
               expected-grid component) and fed through the same race model.

Model (deliberately small: 2026 has 17 races x 22 drivers)
------------------------------------------------------------
Level 1, ratings. For every race k we fit two Plackett-Luce (PL) strength
ratings using only races of the same season *before* k:
  theta_race[d]  from the classified finishing orders,
  theta_quali[d] from the qualifying orders,
each parameterised as team effect + driver-within-team effect
(mu_team + delta_driver) with ridge shrinkage (driver effects shrunk much
harder than team effects, so a driver's rating leans on his team-mate's
results when his own sample is small), and exponential recency weights
(half-life in races) because the 2026 pecking order moved at least twice
(McLaren mid-season, Red Bull's last two wins; see T-010 section 3). DNFs
are *excluded* from the finishing orders: they are modelled separately.

Level 2, outcome model. A second PL over the classified finishers of each
past race with utility
    u = b_race * theta_race + b_grid * (-log grid)          ("grid" mode)
fitted on all earlier 2026 races (each with its own level-1 ratings, so the
coefficients learn how much a rating from k-1 earlier races is worth).
-log(grid) concentrates the advantage at the front, matching the 2026/
2022-26 win-rate-by-slot curve in T-010 4.1. In "prequali" mode a third
1-parameter PL, u_q = a_q * theta_quali, is fitted on past qualifying
orders; each simulation draws a qualifying order from it, uses that as the
grid and then applies the grid-mode race model.

Both PL levels are fitted on *truncated* rankings (first 10 stages for the
ratings, first 3 for the outcome model): the full-order likelihood is
dominated by midfield shuffles and gave a front-row win probability far
below what happens (see PredictorConfig.topk_level2).

Blend. Final probabilities are 0.75 x the PL simulation + 0.25 x a
rank-frequency table (P(win/podium/top10 | grid rank) in grid mode,
| championship rank pre-quali; 2022-25 + earlier 2026 rounds). The weight
was chosen on the 2025 walk-forward; on 2026 the blend beats both of its
parts on win and podium log-loss (T-011 handoff has the table).

DNF component. P(not classified) per driver = beta-binomial shrinkage
(driver -> team -> 2026 field rate -> prior 15%) over earlier 2026 races
(T-045: shrinkage 20/60, plus a grid-slot factor in grid mode; see PredictorConfig).
T-010: 20.9% of 2026 starts are not classified, 6-50% by team, so ignoring
DNFs makes the front-runners' probabilities far too confident.

Probabilities come from Monte Carlo over whole finishing orders (PL is
exactly "utility + Gumbel noise, sort"), so win probabilities sum to 1,
podium to 3 and top-10 to 10 by construction.

Why not XGBoost (CLAUDE.md, Phase 7): ~370 driver-races of 2026 data, and
the targets are rank events that must be coherent across a field. A
per-driver classifier gives win probabilities that don't sum to 1 and needs
far more data to learn the grid curve; PL gets coherence for free with 2-3
coefficients. Revisit once simulator outputs exist as features (Phase 7).

2025 data: only (a) as a down-weighted prior for the level-2 coefficients
(how much grid vs form matters is structural, not car pace) and (b) as a
prior for the within-team driver effect (driver skill). 2025 car/team pace
is never used (T-010 3.1: the order inverted). Both are switched by
PredictorConfig and kept only if they beat 2026-only walk-forward (see
predict_race.py --evaluate and team/handoffs/T-011-data-scientist.md).

Leakage guards: every rating/coefficient/DNF fit for race (s, r) is built
from rows filtered to `round < r` of season s (plus, if enabled, complete
earlier seasons). Of the target race itself only the entry list (driver,
team) and, in grid mode, the `grid` column are read. backend/tests/
test_predictor.py checks that scrambling the target race's results
(except grid) and every later race leaves the prediction unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from scipy.optimize import minimize

PROB_CLIP = 1e-4
# FIA race points by finishing position (P1..P10), 0 beyond; no fastest-lap point since 2025.
FIA_POINTS = np.array([25, 18, 15, 12, 10, 8, 6, 4, 2, 1] + [0] * 12, dtype=float)


@dataclass(frozen=True)
class PredictorConfig:
    # level 1 ratings. Defaults chosen by walk-forward over the 2025 season
    # (an independent season, so 2026 evaluation isn't tuned on itself);
    # see tune_on_season() and the T-011 handoff for the grid that was tried.
    # 2025 walk-forward (rounds 2-24, objective win+podium log-loss, both
    # modes) over halflife {2,4,8,inf} x lambda_team {.05,.2,1,3} x
    # lambda_driver {.2,.5,2,8} x topk_level1 {all,10} x topk_level2 {3,5,10,all}:
    # a broad plateau at weak driver shrinkage / stronger team shrinkage /
    # truncated PL. inf and 8 tie within 0.003; 8 was pre-committed (before
    # looking at 2026) because 2026's order shifted mid-season (T-010 3.1).
    halflife_races: float = 8.0
    lambda_team: float = 1.0
    lambda_driver: float = 0.5
    topk_level1: int | None = 10  # PL stages used for the ratings
    # level 2 outcome model
    lambda_beta: float = 0.05  # weak ridge toward 0 (only matters at round 2-3)
    # Fitting the outcome PL on the full order let the noisy midfield set
    # the noise scale and made the front of the grid under-confident
    # (2026 first pass: P(win|pole) 0.44 vs 0.69 observed). Using only the
    # first 3 selection stages fixes that (2025 tuning: 3 < 5 < 10 < all).
    topk_level2: int | None = 3  # PL stages used for b_race/b_grid
    topk_quali: int | None = 3  # PL stages used for a_quali
    # 2025 priors, kept only if they beat 2026-only in the 2026 walk-forward
    # (python predict_race.py --evaluate). Level-2 coefficient prior: each
    # 2025 race (features built within 2025) enters the b_race/b_grid/a_quali
    # fit with this weight. It helps in both modes, mostly at rounds 2-5
    # where 2026 alone has 1-4 races to fit on -> kept (1.0).
    w2025_level2: float = 1.0
    # Driver-skill prior: delta_driver ridge centre = kappa * the driver's
    # 2025 within-team effect. Mixed result (helps pre-quali by ~0.002
    # log-loss, hurts grid mode by about as much) -> not kept (0.0).
    driver_prior_2025: float = 0.0
    # DNF model: field rate <- prior, team <- field, driver <- team
    dnf_prior_rate: float = 0.15
    dnf_prior_n: float = 22.0
    # T-045 (2026-10-26): shrinkage raised 10/20 -> 20/60 (tuned on 2025, confirmed better on
    # 2026: DNF log-loss 2025 .3503 -> .3434, 2026 .4356 -> .4334; see eval/dnf_audit.py).
    # The gains are small (SE of one season's log-loss is ~.03) but go the same way in both seasons.
    dnf_team_m: float = 20.0
    dnf_driver_m: float = 60.0
    # T-045 grid-slot DNF factor (GRID MODE ONLY; 0 = off). Top-3 starters retired 5.5% of the time in
    # 2022-25 vs 11-13% for the field (fewer first-lap/midfield incidents), 2026 not different enough to
    # reject it. p_dnf *= f_bucket ** dnf_grid_power, f_bucket = (bucket DNF rate in EARLIER seasons,
    # 20-start shrinkage to the overall rate) / overall rate, buckets 1-3 / 4-6 / 7-10 / 11-15 / 16-22,
    # then rescaled so the field's expected DNFs are unchanged. Power .5 beat 1.0 and 0 on 2025
    # (.3425 vs .3432/.3434) and on 2026 (.4307 vs .4310/.4334). Not used pre-quali (no grid).
    dnf_grid_power: float = 0.5
    # T-029 track-level DNF component (OFF by default, 0 = off). p_dnf is multiplied by
    #     ratio_c = (obs_c + m) / (exp_c + m),
    # obs_c = not-classified starts at this circuit in STRICTLY EARLIER races (any
    # season, incl. 2018-21 track_results), exp_c = what the field rate of each of
    # those seasons would have produced; m = dnf_track_m pseudo expected-DNFs of
    # shrinkage toward ratio 1. Season-relative so regulation eras don't leak in.
    dnf_track_m: float = 0.0
    dnf_track_clip: float = 0.9  # upper bound on the adjusted p_dnf
    # Monte Carlo
    n_sims: int = 20000
    seed: int = 0
    # practice long-run pace (OpenF1) as a 3rd level-2 feature
    use_fp: bool = False
    # Final probabilities = (1 - w) * PL model + w * rank-frequency baseline
    # (grid-rank table in grid mode, championship-rank table pre-quali; see
    # RankBaseline). w picked on the 2025 walk-forward before looking at
    # 2026: w in {0, .25, .5, .75, 1}. Caveat (QA sweep): on the 2025 walk-forward
    # WIN log-loss is actually best at w=0; .25 was chosen for podium/top-10
    # log-loss, where the PL model's -log(grid) curve is slightly over-confident
    # for the top-3 starters and the historical table is a well-calibrated
    # shrinkage target.
    blend_baseline: float = 0.25
    # T-040: baseline weight for p_win in GRID mode only (podium/top10 keep
    # blend_baseline). 0 = model only (QA sweep: the grid-rank table adds noise to
    # the winner call; the model's win log-loss is better without it). Set to
    # None to use blend_baseline for win too (pre-T-040 behaviour).
    blend_win_grid: float | None = 0.0
    # T-040: apply per-circuit stickiness in pre-quali mode too. Off: stickiness
    # (T-037) made pre-quali win/winner log-loss worse in both seasons.
    stick_prequali: bool = False
    # T-040: drop non-starters (status "Did not start", "Withdrew", DNQ, ...)
    # from the DNF fit. They never raced, so they say nothing about reliability.
    dnf_exclude_dns: bool = True
    # T-014 per-track grid stickiness (see track_stickiness()). The grid term
    # of the outcome utility is scaled per race by
    #     m = exp(stick_gamma * (s_circuit - s_global)),
    # s = weighted mean Spearman(grid, finish) over earlier races (shrunk to
    # the global mean with stick_k pseudo-races). 0.0 = off (m == 1).
    stick_gamma: float = 4.0  # T-037: with stick_mode="pole" (2025-tuned; 0 = off)
    stick_k: float = 4.0
    stick_w_old: float = 0.5  # weight of races from earlier seasons (era-mixed)
    # T-037 which per-race signal feeds the stickiness: "rho" = whole-field
    # Spearman (T-014); "front" = Spearman over the top stick_top starters only;
    # "pole" = 1 if the winner started P1 else 0 (front-of-grid win lock-in).
    stick_mode: str = "pole"
    stick_top: int = 8
    # races of seasons < 2022 (pre ground-effect) get stick_w_old * this factor
    stick_w_pre2022: float = 1.0


# ----------------------------------------------------------- data shaping


def prepare_history(
    results: pd.DataFrame,
    qualifying: pd.DataFrame,
    sprint: pd.DataFrame | None = None,
    fp: pd.DataFrame | None = None,
    track_results: pd.DataFrame | None = None,
    schedule: pd.DataFrame | None = None,
) -> dict:
    """Light normalisation of the jolpica_client tables. `sprint` (optional)
    only feeds the championship-order baseline/blend (official standings
    include sprint points). `fp` (optional): practice long-run gaps with
    columns season, round, code, fp_gap (s), see openf1_practice.py; only
    used when PredictorConfig.use_fp."""
    res = results.copy()
    res["season"] = res["season"].astype(int)
    res["round"] = res["round"].astype(int)
    res["classified"] = res["classified"].astype(bool)
    q = qualifying.copy()
    if len(q):
        q["season"] = q["season"].astype(int)
        q["round"] = q["round"].astype(int)
    out = {"results": res, "qualifying": q}
    if sprint is not None and len(sprint):
        sp = sprint.copy()
        sp["season"] = sp["season"].astype(int)
        sp["round"] = sp["round"].astype(int)
        out["sprint"] = sp
    if track_results is not None and len(track_results):
        tr = track_results.copy()
        tr["season"] = tr["season"].astype(int)
        tr["round"] = tr["round"].astype(int)
        tr["classified"] = tr["classified"].astype(bool)
        out["track_results"] = tr  # extra seasons used ONLY for track stickiness
    if schedule is not None and len(schedule):
        sc = schedule.copy()
        sc["season"] = sc["season"].astype(int)
        sc["round"] = sc["round"].astype(int)
        out["schedule"] = sc
    if fp is not None and len(fp):
        f = fp.copy()
        f["season"] = f["season"].astype(int)
        f["round"] = f["round"].astype(int)
        out["fp"] = f
    return out


def _grid_eff(grid: pd.Series, n: int) -> np.ndarray:
    """Jolpica grid 0 = pit-lane start (or no grid slot) -> treat as last."""
    g = pd.to_numeric(grid, errors="coerce").fillna(0).to_numpy(dtype=float)
    g[g <= 0] = n
    return np.minimum(g, n)


# ------------------------------------------------- track grid stickiness


def _race_rho(res: pd.DataFrame, mode: str = "rho", top: int = 8) -> pd.DataFrame:
    """Per race: Spearman(grid, finishing position) among classified cars
    (grid 0 / pit lane = last). Columns season, round, circuit_id, rho."""
    rows = []
    for (s, r), g in res.groupby(["season", "round"]):
        n = len(g)
        c = g[g["classified"]]
        if len(c) < 5:
            continue
        ge = _grid_eff(c["grid"], n)
        if mode == "pole":
            w = c[c["position"] == c["position"].min()]
            rho = float(ge[(c["position"] == c["position"].min()).to_numpy()][0] <= 1) if len(w) else np.nan
        elif mode == "front":
            k = ge <= top
            if k.sum() < 4:
                continue
            rho = pd.Series(ge[k]).rank().corr(pd.Series(c["position"].to_numpy(dtype=float)[k]).rank())
        else:
            rho = pd.Series(ge).rank().corr(pd.Series(c["position"].to_numpy(dtype=float)).rank())
        if np.isfinite(rho):
            rows.append((int(s), int(r), str(g["circuit_id"].iloc[0]), float(rho)))
    return pd.DataFrame(rows, columns=["season", "round", "circuit_id", "rho"])


def _rho_table(hist: dict, mode: str = "rho", top: int = 8) -> pd.DataFrame:
    key = ("_rho", mode, top)
    if key not in hist:  # cache on the (mutable) history dict
        parts = [hist["results"]] + ([hist["track_results"]] if "track_results" in hist else [])
        cols = ["season", "round", "circuit_id", "grid", "position", "classified"]
        hist[key] = _race_rho(pd.concat([p[cols] for p in parts], ignore_index=True), mode, top)
    return hist[key]


def circuit_of(hist: dict, season: int, rnd: int) -> str | None:
    res = hist["results"]
    r = res[(res["season"] == season) & (res["round"] == rnd)]
    if len(r):
        return str(r["circuit_id"].iloc[0])
    sc = hist.get("schedule")
    if sc is not None:
        r = sc[(sc["season"] == season) & (sc["round"] == rnd)]
        if len(r):
            return str(r["circuit_id"].iloc[0])
    return None


def track_stickiness(hist: dict, season: int, rnd: int, cfg: "PredictorConfig") -> dict:
    """
    How strongly finishing order follows grid order at the target's circuit,
    from STRICTLY EARLIER races only ((s, r) < (season, rnd), any season in
    results/track_results). Races of the target season weigh 1, earlier
    seasons cfg.stick_w_old (era-mixed: regs changed; overtaking difficulty is
    mostly a track property so they are down-weighted, not dropped).
        s_c = (sum w*rho_c + k*s_global) / (sum w_c + k)
    shrunk to the global weighted mean. Unknown circuit (e.g. Madrid) or no
    history -> s_c = s_global -> multiplier exactly 1.
    Returns dict(mult, s_circuit, s_global, n_cur, n_old, circuit_id).
    """
    t = _rho_table(hist, cfg.stick_mode, cfg.stick_top)
    t = t[(t["season"] < season) | ((t["season"] == season) & (t["round"] < rnd))]
    cid = circuit_of(hist, season, rnd)
    if t.empty:
        return dict(mult=1.0, s_circuit=np.nan, s_global=np.nan, n_cur=0, n_old=0, circuit_id=cid)
    w = np.where(t["season"] == season, 1.0,
                 np.where(t["season"] < 2022, cfg.stick_w_old * cfg.stick_w_pre2022, cfg.stick_w_old))
    s_glob = float((w * t["rho"]).sum() / w.sum())
    m = (t["circuit_id"] == cid).to_numpy()
    sw = float(w[m].sum())
    s_c = (float((w[m] * t["rho"].to_numpy()[m]).sum()) + cfg.stick_k * s_glob) / (sw + cfg.stick_k)
    mult = float(np.exp(cfg.stick_gamma * (s_c - s_glob)))
    return dict(mult=mult, s_circuit=s_c, s_global=s_glob,
                n_cur=int((m & (t["season"] == season).to_numpy()).sum()),
                n_old=int((m & (t["season"] != season).to_numpy()).sum()), circuit_id=cid)


# ------------------------------------------------------ Plackett-Luce core


def pl_loglik_grad(u: np.ndarray, topk: int | None = None) -> tuple[float, np.ndarray]:
    """
    Log-likelihood of an observed ranking (u given in finishing order,
    best first) under PL, and its gradient w.r.t. u:
        ll = sum_{i<=K} [u_i - log D_i],   D_i = sum_{j>=i} exp(u_j)
        dll/du_j = 1[j<=K] - exp(u_j) * sum_{i<=min(j,K)} 1/D_i
    `topk` = K truncates the likelihood to the first K selection stages
    ("who won, then who was 2nd of the rest, ...") while every lower
    finisher still sits in the denominators. See PredictorConfig.topk_*.
    """
    n = len(u)
    if n < 2:
        return 0.0, np.zeros_like(u)
    K = n if topk is None else min(topk, n)
    m = u.max()
    e = np.exp(u - m)
    D = np.cumsum(e[::-1])[::-1]
    stage = np.arange(n) < K
    ll = float(np.sum(((u - m) - np.log(D))[stage]))
    inv = np.where(stage, 1.0 / D, 0.0)
    g = stage.astype(float) - e * np.cumsum(inv)
    return ll, g


def fit_pl(designs, weights, penalty, center=None, topk: int | None = None) -> np.ndarray:
    """
    Weighted, ridge-penalised PL fit. designs: list of (n_i x p) matrices
    with rows in finishing order; weights: per-ranking weight; penalty:
    length-p ridge strengths; center: ridge centre (default 0).
    """
    p = len(penalty)
    penalty = np.asarray(penalty, dtype=float)
    center = np.zeros(p) if center is None else np.asarray(center, dtype=float)
    if not designs:
        return center.copy()

    def f(x):
        nll = 0.5 * np.sum(penalty * (x - center) ** 2)
        grad = penalty * (x - center)
        for X, w in zip(designs, weights):
            ll, g = pl_loglik_grad(X @ x, topk)
            nll -= w * ll
            grad -= w * (X.T @ g)
        return nll, grad

    out = minimize(f, center.copy(), jac=True, method="L-BFGS-B")
    return out.x


# ---------------------------------------------------------- level 1: ratings


@dataclass
class Ratings:
    team: dict = field(default_factory=dict)
    driver: dict = field(default_factory=dict)
    driver_center: dict = field(default_factory=dict)

    def of(self, driver_id: str, team_id: str) -> float:
        d = self.driver.get(driver_id, self.driver_center.get(driver_id, 0.0))
        return self.team.get(team_id, 0.0) + d


def fit_ratings(
    orders: list[pd.DataFrame],
    ages: list[float],
    cfg: PredictorConfig,
    driver_center: dict | None = None,
    halflife: float | None = None,
) -> Ratings:
    """
    orders: one DataFrame per past session (driver_id, team_id) sorted best
    first, classified rows only. ages: races before the target (0 = the
    most recent one). Returns team + driver-within-team PL strengths.
    """
    driver_center = driver_center or {}
    hl = cfg.halflife_races if halflife is None else halflife
    teams = sorted({t for o in orders for t in o["team_id"]})
    drivers = sorted({d for o in orders for d in o["driver_id"]})
    if not orders:
        return Ratings(driver_center=dict(driver_center))
    ti = {t: i for i, t in enumerate(teams)}
    di = {d: len(teams) + i for i, d in enumerate(drivers)}
    p = len(teams) + len(drivers)
    designs = []
    for o in orders:
        X = np.zeros((len(o), p))
        X[np.arange(len(o)), [ti[t] for t in o["team_id"]]] = 1.0
        X[np.arange(len(o)), [di[d] for d in o["driver_id"]]] = 1.0
        designs.append(X)
    weights = [0.5 ** (a / hl) if np.isfinite(hl) else 1.0 for a in ages]
    penalty = np.r_[np.full(len(teams), cfg.lambda_team), np.full(len(drivers), cfg.lambda_driver)]
    center = np.r_[np.zeros(len(teams)), [driver_center.get(d, 0.0) for d in drivers]]
    x = fit_pl(designs, weights, penalty, center, cfg.topk_level1)
    return Ratings(
        team={t: x[ti[t]] for t in teams},
        driver={d: x[di[d]] for d in drivers},
        driver_center=dict(driver_center),
    )


def _past_orders(hist: dict, season: int, rnd: int, kind: str):
    """Classified orders of earlier rounds in the same season, oldest first."""
    if kind == "race":
        df = hist["results"]
        df = df[(df["season"] == season) & (df["round"] < rnd) & df["classified"]]
        key = "position"
    else:
        df = hist["qualifying"]
        if len(df) == 0:
            return [], []
        df = df[(df["season"] == season) & (df["round"] < rnd)].dropna(subset=["quali_position"])
        key = "quali_position"
    rounds = sorted(df["round"].unique())
    orders, ages = [], []
    for k in rounds:
        o = df[df["round"] == k].sort_values(key)
        orders.append(o[["driver_id", "team_id"]].reset_index(drop=True))
        ages.append(float(len(rounds) - 1 - rounds.index(k)))
    return orders, ages


def driver_prior_from_season(hist: dict, season: int, cfg: PredictorConfig) -> dict:
    """Within-team driver effects from a whole earlier season (no decay)."""
    df = hist["results"]
    last = df[df["season"] == season]["round"].max()
    if pd.isna(last):
        return {}
    orders, ages = _past_orders(hist, season, int(last) + 1, "race")
    r = fit_ratings(orders, ages, cfg, halflife=np.inf)
    return r.driver


# ---------------------------------------------------- per-race feature rows


def entrants_for(hist: dict, season: int, rnd: int, entry_list: pd.DataFrame | None = None):
    """
    Entry list (driver_id, code, team_id, team) for the target race. Uses,
    in order: an explicit entry_list, the race's own results (entry list
    only), that round's qualifying, else the latest earlier round's grid.
    """
    if entry_list is not None:
        return entry_list.reset_index(drop=True)
    cols = ["driver_id", "code", "team_id", "team"]
    res = hist["results"]
    r = res[(res["season"] == season) & (res["round"] == rnd)]
    if len(r):
        return r[cols].reset_index(drop=True)
    q = hist["qualifying"]
    if len(q):
        qq = q[(q["season"] == season) & (q["round"] == rnd)]
        if len(qq):
            return qq[cols].reset_index(drop=True)
    prev = res[(res["season"] == season) & (res["round"] < rnd)]
    last = prev["round"].max()
    return prev[prev["round"] == last][cols].reset_index(drop=True)


def race_features(hist, season, rnd, cfg, driver_center=None, entry_list=None, grid=None):
    """
    Feature rows for every entrant of (season, rnd), built from earlier
    rounds of that season only. `grid` (Series aligned to entrants, or None)
    overrides the race's own grid column.
    """
    ent = entrants_for(hist, season, rnd, entry_list).copy()
    o_r, a_r = _past_orders(hist, season, rnd, "race")
    o_q, a_q = _past_orders(hist, season, rnd, "quali")
    rr = fit_ratings(o_r, a_r, cfg, driver_center)
    rq = fit_ratings(o_q, a_q, cfg, driver_center)
    ent["theta_race"] = [rr.of(d, t) for d, t in zip(ent["driver_id"], ent["team_id"])]
    ent["theta_quali"] = [rq.of(d, t) for d, t in zip(ent["driver_id"], ent["team_id"])]
    n = len(ent)
    if grid is not None:
        ent["grid"] = np.asarray(grid)
    else:
        res = hist["results"]
        r = res[(res["season"] == season) & (res["round"] == rnd)]
        if len(r):
            ent["grid"] = ent["driver_id"].map(dict(zip(r["driver_id"], r["grid"])))
        else:
            ent["grid"] = np.nan
    ent["grid_eff"] = _grid_eff(ent["grid"], n) if ent["grid"].notna().any() else np.nan
    ent["fp_feat"] = _fp_feature(hist, season, rnd, ent)
    ent["stick"] = track_stickiness(hist, season, rnd, cfg)["mult"] if cfg.stick_gamma else 1.0
    ent["season"], ent["round"] = season, rnd
    return ent


def _fp_feature(hist, season, rnd, ent) -> np.ndarray:
    """-(team long-run gap in s) from this weekend's practice (available
    before qualifying). Teams without a long run get the field median; a
    weekend without data gets 0 for everyone (no information)."""
    fp = hist.get("fp")
    if fp is None:
        return np.zeros(len(ent))
    f = fp[(fp["season"] == season) & (fp["round"] == rnd)]
    if f.empty:
        return np.zeros(len(ent))
    gap = ent["code"].map(dict(zip(f["code"], f["fp_gap"])))
    team_gap = gap.groupby(ent["team_id"]).transform("mean")
    team_gap = team_gap.fillna(team_gap.median())
    return -(team_gap - team_gap.min()).to_numpy(dtype=float)


def _outcomes(hist, season, rnd) -> pd.DataFrame:
    res = hist["results"]
    r = res[(res["season"] == season) & (res["round"] == rnd)]
    return r[["driver_id", "position", "classified"]]


class FeatureStore:
    """Caches race_features per (season, round) for one config/prior."""

    def __init__(self, hist, cfg, driver_center=None):
        self.hist, self.cfg, self.dc = hist, cfg, driver_center
        self._c = {}

    def get(self, season, rnd):
        k = (season, rnd)
        if k not in self._c:
            f = race_features(self.hist, season, rnd, self.cfg, self.dc)
            o = _outcomes(self.hist, season, rnd)
            f = f.merge(o, on="driver_id", how="left")
            self._c[k] = f
        return self._c[k]


# ------------------------------------------------- level 2 + DNF fitting


@dataclass
class OutcomeModel:
    b_race: float
    b_grid: float
    a_quali: float
    b_fp: float = 0.0


def _training_races(hist, season, rnd, cfg):
    res = hist["results"]
    rows = [(season, k, 1.0) for k in sorted(res[(res["season"] == season) & (res["round"] < rnd)]["round"].unique())]
    if cfg.w2025_level2 > 0 and season > 2025:  # never a season's own (future) races
        for k in sorted(res[res["season"] == 2025]["round"].unique()):
            rows.append((2025, k, cfg.w2025_level2))
    return rows


def fit_outcome_model(hist, season, rnd, cfg, stores: dict) -> OutcomeModel:
    """Fit b_race, b_grid (race orders) and a_quali (quali orders) on
    earlier races only. `stores` maps season -> FeatureStore."""
    designs, weights, qdesigns, qweights = [], [], [], []
    q = hist["qualifying"]
    for s, k, w in _training_races(hist, season, rnd, cfg):
        f = stores[s].get(s, k)
        fin = f[f["classified"] == True].sort_values("position")  # noqa: E712
        if len(fin) >= 2:
            X = np.c_[fin["theta_race"].to_numpy(),
                      fin["stick"].to_numpy() * -np.log(fin["grid_eff"].to_numpy(dtype=float)),
                      fin["fp_feat"].to_numpy()]
            if not cfg.use_fp:
                X = X[:, :2]
            designs.append(X)
            weights.append(w)
        if len(q):
            qq = q[(q["season"] == s) & (q["round"] == k)].dropna(subset=["quali_position"])
            qq = qq.sort_values("quali_position").merge(f[["driver_id", "theta_quali"]], on="driver_id")
            if len(qq) >= 2:
                qdesigns.append(qq[["theta_quali"]].to_numpy())
                qweights.append(w)
    b = fit_pl(designs, weights, [cfg.lambda_beta] * (3 if cfg.use_fp else 2), topk=cfg.topk_level2)
    a = fit_pl(qdesigns, qweights, [cfg.lambda_beta], topk=cfg.topk_quali)
    return OutcomeModel(b_race=float(b[0]), b_grid=float(b[1]), a_quali=float(a[0]),
                        b_fp=float(b[2]) if cfg.use_fp else 0.0)


def track_dnf_ratio(hist: dict, season: int, rnd: int, cfg: "PredictorConfig") -> float:
    """Shrunk circuit DNF multiplier (see PredictorConfig.dnf_track_m); 1.0 when off,
    when the circuit is unknown or has no earlier race. Only races strictly before
    (season, rnd) are read; a past season's field rate uses that whole season, the
    target season's uses its earlier rounds only."""
    if not cfg.dnf_track_m:
        return 1.0
    cid = circuit_of(hist, season, rnd)
    if cid is None:
        return 1.0
    parts = [hist["results"]] + ([hist["track_results"]] if "track_results" in hist else [])
    cols = ["season", "round", "circuit_id", "classified"]
    a = pd.concat([p[cols] for p in parts], ignore_index=True)
    a = a[(a["season"] < season) | ((a["season"] == season) & (a["round"] < rnd))]
    a = a.assign(nf=(~a["classified"].astype(bool)).astype(float))
    rate = a.groupby("season")["nf"].mean()
    c = a[a["circuit_id"] == cid]
    if c.empty:
        return 1.0
    obs = float(c["nf"].sum())
    exp = float(c["season"].map(rate).sum())
    return (obs + cfg.dnf_track_m) / (exp + cfg.dnf_track_m)


DNS_STATUS = frozenset({"did not start", "withdrew", "did not qualify", "dnq", "excluded", "did not participate"})


def _started(df: pd.DataFrame) -> pd.Series:
    """False for entries that never started (see DNS_STATUS). A lap-1 'Retired'
    with 0 laps did start and is a real DNF, so laps is deliberately not used."""
    return ~df["status"].fillna("").str.strip().str.lower().isin(DNS_STATUS)


GRID_DNF_BINS = [0, 3, 6, 10, 15, 22]


def grid_dnf_factors(hist, season, cfg) -> np.ndarray | None:
    """Per grid-bucket DNF multiplier from seasons strictly before `season` (see
    PredictorConfig.dnf_grid_power); None if there is no earlier season in `hist`."""
    res = hist["results"]
    h = res[res["season"] < season]
    if h.empty:
        return None
    if cfg.dnf_exclude_dns and "status" in h.columns:
        h = h[_started(h)]
    nf = (~h["classified"]).astype(float)
    b = pd.cut(pd.to_numeric(h["grid"], errors="coerce").fillna(0).replace(0, 22).clip(upper=22), GRID_DNF_BINS, labels=False)
    ov = nf.mean()
    g = nf.groupby(b).agg(["sum", "count"]).reindex(range(len(GRID_DNF_BINS) - 1), fill_value=0)
    return (((g["sum"] + 20 * ov) / (g["count"] + 20)) / ov).to_numpy()


def dnf_probabilities(hist, season, rnd, entrants: pd.DataFrame, cfg, use_grid: bool = False) -> np.ndarray:
    """Shrunk P(not classified) for each entrant from earlier rounds of the
    season: field <- prior, team <- field, driver <- team. use_grid (grid mode
    only): scale by the starting-slot factor of grid_dnf_factors()."""
    res = hist["results"]
    past = res[(res["season"] == season) & (res["round"] < rnd)]
    if cfg.dnf_exclude_dns and "status" in past.columns:
        past = past[_started(past)]
    nf = (~past["classified"]).astype(float)
    field_rate = (nf.sum() + cfg.dnf_prior_rate * cfg.dnf_prior_n) / (len(past) + cfg.dnf_prior_n)
    t_n = past.groupby("team_id").size()
    t_k = nf.groupby(past["team_id"]).sum()
    d_n = past.groupby("driver_id").size()
    d_k = nf.groupby(past["driver_id"]).sum()
    out = []
    for d, t in zip(entrants["driver_id"], entrants["team_id"]):
        tr = (t_k.get(t, 0.0) + cfg.dnf_team_m * field_rate) / (t_n.get(t, 0) + cfg.dnf_team_m)
        dr = (d_k.get(d, 0.0) + cfg.dnf_driver_m * tr) / (d_n.get(d, 0) + cfg.dnf_driver_m)
        out.append(dr)
    out = np.asarray(out)
    if use_grid and cfg.dnf_grid_power and "grid_eff" in entrants.columns and entrants["grid_eff"].notna().all():
        fac = grid_dnf_factors(hist, season, cfg)
        if fac is not None:
            b = pd.cut(entrants["grid_eff"].to_numpy(dtype=float), GRID_DNF_BINS, labels=False)
            adj = out * fac[b] ** cfg.dnf_grid_power
            out = adj * out.sum() / adj.sum()
    if cfg.dnf_track_m:
        out = np.minimum(out * track_dnf_ratio(hist, season, rnd, cfg), cfg.dnf_track_clip)
    return out


# ------------------------------------------------------------- simulation


def simulate(
    theta_race, theta_quali, grid_eff, p_dnf, model: OutcomeModel, mode: str, n_sims: int, seed: int,
    fp_feat=None, stick=1.0,
) -> dict:
    """
    Monte Carlo over finishing orders. Returns per-driver arrays p_win,
    p_podium, p_top10, exp_pos and (prequali) exp_grid.
    """
    rng = np.random.default_rng(seed)
    n = len(theta_race)
    if mode == "grid":
        g = np.broadcast_to(grid_eff, (n_sims, n))
    elif mode == "prequali":
        uq = model.a_quali * np.asarray(theta_quali)
        perf_q = uq + rng.gumbel(size=(n_sims, n))
        # rank 1 = best quali performance
        g = np.empty((n_sims, n))
        order = np.argsort(-perf_q, axis=1)
        g[np.arange(n_sims)[:, None], order] = np.arange(1, n + 1)
    else:
        raise ValueError(mode)
    u = model.b_race * np.asarray(theta_race) + model.b_grid * stick * (-np.log(g))
    if fp_feat is not None and model.b_fp:
        u = u + model.b_fp * np.asarray(fp_feat)
    perf = u + rng.gumbel(size=(n_sims, n))
    dnf = rng.random((n_sims, n)) < np.asarray(p_dnf)
    perf = np.where(dnf, perf - 1e6, perf)  # DNFs rank behind every finisher
    order = np.argsort(-perf, axis=1)
    pos = np.empty((n_sims, n))
    pos[np.arange(n_sims)[:, None], order] = np.arange(1, n + 1)
    out = {
        "p_win": (pos == 1).mean(0),
        "p_podium": (pos <= 3).mean(0),
        "p_top10": (pos <= 10).mean(0),
        "exp_pos": pos.mean(0),
        "med_pos": np.median(pos, axis=0),
        # T-045 ranking statistics: expected position given classified, expected FIA points
        "exp_pos_cls": np.array([pos[~dnf[:, j], j].mean() if (~dnf[:, j]).any() else float(n) for j in range(n)]),
        "exp_pts": FIA_POINTS[np.minimum(pos.astype(int), len(FIA_POINTS)) - 1].mean(0),
        "p_dnf_sim": dnf.mean(0),
        "exp_grid": g.mean(0),
    }
    if mode == "prequali":  # T-035: expose the simulated qualifying distribution
        out.update(quali_events(g))
    return out


def rank_by_points(exp_pts, exp_pos) -> np.ndarray:
    """1-based predicted finishing order: highest expected FIA points first, expected position as the
    tie-break (the back of the grid, where expected points are ~0 for everyone).

    Why this and not the mean position (T-045, 2026-10-26): a DNF puts a car ~18th-20th, so a >50% win
    favourite with a 15-18% DNF chance has a mean position of 3-4.5 and was listed below drivers it beats
    in every other statistic (Dutch GP 2026: NOR p_win .78 behind ANT p_win .05). Expected points is
    linear in the finishing distribution like the mean position, but weights the front by the points
    scale (25-18-15...), so a 25-point win outweighs a 0-point DNF tail rather than a 20-place one.
    Over 41 walk-forward races (2025 R2-24, 2026 R1-18) it had the best rank correlation and MAE of the
    candidates and zero favourite/podium inversions; see data/jolpica/eval/rank_stat_eval.py."""
    exp_pts, exp_pos = np.asarray(exp_pts, dtype=float), np.asarray(exp_pos, dtype=float)
    order = np.lexsort((exp_pos, -exp_pts))
    rank = np.empty(len(order), dtype=int)
    rank[order] = np.arange(1, len(order) + 1)
    return rank


def q1_cutoff(n: int) -> int:
    """First grid slot that is eliminated in Q1 under the current format.
    22 cars: 16 advance, 6 out (slots 17-22). 20 cars (<=2025 era): 15 advance, 5 out.
    Smaller fields: the same 15 advance rule, nobody out if n <= 15."""
    advance = 16 if n >= 21 else 15
    return min(advance, n) + 1


def quali_events(g: np.ndarray) -> dict:
    """Per-driver probabilities from simulated qualifying orders g (n_sims, n), rank 1 = pole.
    p_q3 = top 10, p_q1_exit = slot >= q1_cutoff(n). Qualifying classification, not the
    penalised grid."""
    n = g.shape[1]
    return {
        "p_pole": (g == 1).mean(0),
        "p_front_row": (g <= 2).mean(0),
        "p_q3": (g <= 10).mean(0),
        "p_q1_exit": (g >= q1_cutoff(n)).mean(0),
    }


# ------------------------------------------------------------ public API


def _stores_for(hist, cfg, season):
    dc = None
    if cfg.driver_prior_2025 > 0:
        prior = driver_prior_from_season(hist, 2025, cfg)
        dc = {d: cfg.driver_prior_2025 * v for d, v in prior.items()}
    stores = {season: FeatureStore(hist, cfg, dc)}
    if cfg.w2025_level2 > 0 and season != 2025:
        stores[2025] = FeatureStore(hist, cfg, None)
    return stores


def predict_race(
    hist: dict,
    season: int,
    rnd: int,
    mode: str = "prequali",
    cfg: PredictorConfig = PredictorConfig(),
    entry_list: pd.DataFrame | None = None,
    grid: pd.Series | np.ndarray | None = None,
    stores: dict | None = None,
) -> pd.DataFrame:
    """
    Probability table for race (season, rnd). In "grid" mode the grid comes
    from `grid` (aligned with the entry list) or else the race's own
    Jolpica `grid` column; it must exist.
    """
    if mode == "prequali" and cfg.stick_gamma and not cfg.stick_prequali:
        # stickiness off in pre-quali: refit level 2 without it too (a caller's
        # stores were built with stickiness, so rebuild them)
        cfg = replace(cfg, stick_gamma=0.0)
        stores = None
    stores = stores or _stores_for(hist, cfg, season)
    if entry_list is None and grid is None:
        f = stores[season].get(season, rnd).copy()
    else:
        f = race_features(hist, season, rnd, cfg, stores[season].dc, entry_list, grid)
    if mode == "grid" and f["grid_eff"].isna().any():
        raise ValueError("grid mode needs a grid for every entrant")
    model = fit_outcome_model(hist, season, rnd, cfg, stores)
    p_dnf = dnf_probabilities(hist, season, rnd, f, cfg, use_grid=(mode == "grid"))
    sim = simulate(
        f["theta_race"].to_numpy(),
        f["theta_quali"].to_numpy(),
        f["grid_eff"].to_numpy(dtype=float) if mode == "grid" else None,
        p_dnf,
        model,
        mode,
        cfg.n_sims,
        cfg.seed,
        fp_feat=f["fp_feat"].to_numpy(),
        stick=float(f["stick"].iloc[0]),
    )
    out = f[["driver_id", "code", "team_id", "team", "theta_race", "theta_quali", "grid", "fp_feat"]].copy()
    out["p_dnf"] = p_dnf
    for k in ("p_win", "p_podium", "p_top10", "exp_pos", "med_pos", "exp_pos_cls", "exp_pts"):
        out[k] = sim[k]
    w = cfg.blend_baseline
    if w > 0:
        kind = "grid" if mode == "grid" else "champ"
        ent = f[["driver_id", "grid"]]
        bl = stores.setdefault(f"_baseline_{kind}", RankBaseline(hist, kind))
        bl = bl.predict(season, rnd, entrants=ent).set_index("driver_id")
        for ev in ("win", "podium", "top10"):
            we = cfg.blend_win_grid if (ev == "win" and mode == "grid" and cfg.blend_win_grid is not None) else w
            out[f"p_{ev}_model"] = out[f"p_{ev}"]
            out[f"p_{ev}_baseline"] = out["driver_id"].map(bl[f"p_{ev}"]).to_numpy()
            out[f"p_{ev}"] = (1 - we) * out[f"p_{ev}_model"] + we * out[f"p_{ev}_baseline"]
    out["pred_rank"] = rank_by_points(out["exp_pts"], out["exp_pos"])
    if mode == "prequali":
        out["exp_grid"] = sim["exp_grid"]
        for k in ("p_pole", "p_front_row", "p_q3", "p_q1_exit"):
            out[k] = sim[k]
    out["season"], out["round"], out["mode"] = season, rnd, mode
    out.attrs["model"] = model
    out.attrs["stickiness"] = track_stickiness(hist, season, rnd, cfg)
    return out.sort_values(["p_win", "p_podium"], ascending=False).reset_index(drop=True)


# ------------------------------------------------------------- baselines


def _ranking_scores(hist, season, rnd, kind, entrants=None) -> pd.DataFrame:
    """
    Pre-race ordering of the race's entrants for the rank baselines; lower
    `score` = better. grid: post-penalty grid. champ: cumulative points
    (race + sprint) from earlier rounds. form: mean finishing position over
    the last 3 earlier rounds (unclassified drivers keep Jolpica's
    end-of-field position, i.e. DNF ~ last), unknown drivers = 11.
    """
    res = hist["results"]
    if entrants is None:
        r = res[(res["season"] == season) & (res["round"] == rnd)][["driver_id", "grid"]].copy()
    else:
        r = entrants[["driver_id", "grid"]].copy()
    n = len(r)
    past = res[(res["season"] == season) & (res["round"] < rnd)]
    pts = past.groupby("driver_id")["points"].sum()
    sprint = hist.get("sprint")
    if sprint is not None and len(sprint):
        sp = sprint[(sprint["season"] == season) & (sprint["round"] < rnd)]
        pts = pts.add(sp.groupby("driver_id")["points"].sum(), fill_value=0)
    if kind == "grid":
        r["score"] = _grid_eff(r["grid"], n)
        r["tieb"] = 0
    elif kind == "champ":
        r["score"] = -r["driver_id"].map(pts).fillna(0)
        r["tieb"] = _grid_eff(r["grid"], n) * 0  # no grid info pre-quali
    elif kind == "form":
        last3 = sorted(past["round"].unique())[-3:]
        f = past[past["round"].isin(last3)].groupby("driver_id")["position"].mean()
        r["score"] = r["driver_id"].map(f).fillna(11.0)
        r["tieb"] = -r["driver_id"].map(pts).fillna(0)
    else:
        raise ValueError(kind)
    r = r.sort_values(["score", "tieb", "driver_id"]).reset_index(drop=True)
    r["rank"] = np.arange(1, n + 1)
    return r


class RankBaseline:
    """
    P(event | pre-race rank) frequency tables. Tables are estimated
    walk-forward: complete seasons 2022-2025 plus the target season's
    earlier rounds (rounds >= 2 only, so 'champ'/'form' are defined), with
    add-alpha smoothing; probabilities are then normalised within the race
    (win -> 1, podium -> 3, top10 -> 10). Generous to the baselines: they
    get 4 extra seasons of history the model doesn't use.
    """

    def __init__(self, hist, kind, table_seasons=(2022, 2023, 2024, 2025)):
        self.hist, self.kind = hist, kind
        self.table_seasons = table_seasons
        self._rows = {}

    def _rows_for(self, season, rnd):
        k = (season, rnd)
        if k not in self._rows:
            r = _ranking_scores(self.hist, season, rnd, self.kind)
            o = _outcomes(self.hist, season, rnd)
            self._rows[k] = r.merge(o, on="driver_id")
        return self._rows[k]

    def table(self, season, rnd, max_rank=22, alpha=1.0):
        res = self.hist["results"]
        keys = []
        for s in self.table_seasons:
            if s >= season:
                continue
            keys += [(s, k) for k in sorted(res[res["season"] == s]["round"].unique()) if k >= 2]
        keys += [(season, k) for k in sorted(res[(res["season"] == season) & (res["round"] < rnd)]["round"].unique()) if k >= 2]
        rows = pd.concat([self._rows_for(*k) for k in keys], ignore_index=True)
        tab = {}
        for ev, thr, base in (("win", 1, 1 / 21), ("podium", 3, 3 / 21), ("top10", 10, 10 / 21)):
            y = (rows["position"] <= thr).astype(float)
            cnt = rows.groupby("rank").size().reindex(range(1, max_rank + 1), fill_value=0)
            hit = y.groupby(rows["rank"]).sum().reindex(range(1, max_rank + 1), fill_value=0)
            # smooth toward a monotone-ish fallback: the overall base rate
            tab[ev] = ((hit + alpha * base * 0.2) / (cnt + alpha)).to_numpy()
        return tab

    def predict(self, season, rnd, entrants: pd.DataFrame | None = None) -> pd.DataFrame:
        """entrants (driver_id, grid) for a race without results yet."""
        if entrants is None:
            r = self._rows_for(season, rnd)[["driver_id", "rank"]].copy()
        else:
            r = _ranking_scores(self.hist, season, rnd, self.kind, entrants)[["driver_id", "rank"]].copy()
        tab = self.table(season, rnd)
        idx = np.minimum(r["rank"].to_numpy(), 22) - 1
        for ev, tot in (("win", 1), ("podium", 3), ("top10", 10)):
            p = tab[ev][idx]
            p = p / p.sum() * tot
            # no clipping here (it would break the sums); scores clip later.
            # min(.,1) only binds for tiny synthetic fields (top10 of 10 cars)
            r[f"p_{ev}"] = np.minimum(p, 1.0)
        return r


# ------------------------------------------------------------- evaluation


def _attach_truth(pred: pd.DataFrame, hist, season, rnd) -> pd.DataFrame:
    o = _outcomes(hist, season, rnd)
    p = pred.merge(o, on="driver_id", how="left")
    p["y_win"] = (p["position"] == 1).astype(int)
    p["y_podium"] = (p["position"] <= 3).astype(int)
    p["y_top10"] = (p["position"] <= 10).astype(int)
    p["season"], p["round"] = season, rnd
    return p


def walk_forward(hist, season, rounds, method, cfg=PredictorConfig()) -> pd.DataFrame:
    """
    Predictions for each round in `rounds`, each made from strictly earlier
    information. method: "model_grid", "model_prequali", or a baseline
    "grid" / "champ" / "form" / "uniform".
    """
    out = []
    if method.startswith("model_"):
        mode = method.split("_", 1)[1]
        if mode == "prequali" and not cfg.stick_prequali:
            cfg = replace(cfg, stick_gamma=0.0)  # same as predict_race, but keeps the store cache
        stores = _stores_for(hist, cfg, season)
        for rnd in rounds:
            p = predict_race(hist, season, rnd, mode, cfg, stores=stores)
            out.append(_attach_truth(p[["driver_id", "p_win", "p_podium", "p_top10", "p_dnf"]], hist, season, rnd))
    elif method == "uniform":
        res = hist["results"]
        for rnd in rounds:
            r = res[(res["season"] == season) & (res["round"] == rnd)][["driver_id"]].copy()
            n = len(r)
            r["p_win"], r["p_podium"], r["p_top10"] = 1 / n, 3 / n, 10 / n
            out.append(_attach_truth(r, hist, season, rnd))
    else:
        bl = RankBaseline(hist, method)
        for rnd in rounds:
            out.append(_attach_truth(bl.predict(season, rnd), hist, season, rnd))
    df = pd.concat(out, ignore_index=True)
    for ev in ("win", "podium", "top10"):
        df[f"p_{ev}"] = np.clip(df[f"p_{ev}"], PROB_CLIP, 1 - PROB_CLIP)
    return df


def score(pred: pd.DataFrame) -> dict:
    """Mean per-race binary log-loss and Brier for win/podium/top10, plus
    the multiclass winner log-loss (-log P(actual winner))."""
    out = {}
    for ev in ("win", "podium", "top10"):
        p, y = pred[f"p_{ev}"], pred[f"y_{ev}"]
        ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        br = (p - y) ** 2
        out[f"{ev}_logloss"] = ll.groupby(pred["round"]).mean().mean()
        out[f"{ev}_brier"] = br.groupby(pred["round"]).mean().mean()
    w = pred[pred["y_win"] == 1]
    out["winner_nll"] = float(-np.log(w["p_win"]).mean())
    out["n_races"] = pred["round"].nunique()
    return out


def calibration_table(pred: pd.DataFrame, ev: str, bins=(0, 0.02, 0.05, 0.15, 0.35, 0.65, 1.0)) -> pd.DataFrame:
    b = pd.cut(pred[f"p_{ev}"], bins=list(bins), include_lowest=True)
    g = pred.groupby(b, observed=True)
    return pd.DataFrame(
        {
            "n": g.size(),
            "mean_pred": g[f"p_{ev}"].mean(),
            "observed": g[f"y_{ev}"].mean(),
            "events_pred": g[f"p_{ev}"].sum(),
            "events_obs": g[f"y_{ev}"].sum(),
        }
    )


# ------------------------------------------- qualifying evaluation (T-035)

QUALI_BASELINES = ("prev_quali", "champ", "uniform")


def _quali_truth(hist, season, rnd) -> pd.DataFrame:
    q = hist["qualifying"]
    t = q[(q["season"] == season) & (q["round"] == rnd)].dropna(subset=["quali_position"])
    return t[["driver_id", "quali_position"]].reset_index(drop=True)


def _quali_baseline_rank(hist, season, rnd, kind: str, drivers: pd.Series) -> np.ndarray:
    """1-based pre-qualifying rank of `drivers` by a simple rule, from strictly earlier rounds.
    prev_quali: qualifying position at the latest earlier round of the season that has one
    (unseen drivers last). champ: points of earlier rounds (race + sprint)."""
    if kind == "prev_quali":
        q = hist["qualifying"]
        prev = q[(q["season"] == season) & (q["round"] < rnd)].dropna(subset=["quali_position"])
        score = pd.Series(99.0, index=drivers.index)
        if len(prev):
            last = prev[prev["round"] == prev["round"].max()]
            score = drivers.map(dict(zip(last["driver_id"], last["quali_position"]))).fillna(99.0)
    else:
        res = hist["results"]
        pts = res[(res["season"] == season) & (res["round"] < rnd)].groupby("driver_id")["points"].sum()
        sp = hist.get("sprint")
        if sp is not None and len(sp):
            pts = pts.add(sp[(sp["season"] == season) & (sp["round"] < rnd)].groupby("driver_id")["points"].sum(), fill_value=0)
        score = -drivers.map(pts).fillna(0.0)
    order = pd.DataFrame({"s": score.to_numpy(), "d": drivers.to_numpy()}).sort_values(["s", "d"]).index
    rank = np.empty(len(drivers))
    rank[order] = np.arange(1, len(drivers) + 1)
    return rank


def _quali_rank_table(hist, season, rnd, kind: str, alpha=1.0) -> dict:
    """P(event | baseline rank) from 2022-2025 plus earlier rounds (>=2) of `season`,
    add-alpha smoothed. Same generosity as RankBaseline: 4 extra seasons of history."""
    q = hist["qualifying"]
    keys = []
    for s in (2022, 2023, 2024, 2025):
        if s < season:
            keys += [(s, k) for k in sorted(q[q["season"] == s]["round"].unique()) if k >= 2]
    keys += [(season, k) for k in sorted(q[(q["season"] == season) & (q["round"] < rnd)]["round"].unique()) if k >= 2]
    rows = []
    for s, k in keys:
        t = _quali_truth(hist, s, k)
        t["rank"] = _quali_baseline_rank(hist, s, k, kind, t["driver_id"])
        t["n"] = len(t)
        rows.append(t)
    r = pd.concat(rows, ignore_index=True)
    out = {}
    for ev, thr, prior in (("pole", 1, 1 / 21), ("front_row", 2, 2 / 21), ("q3", 10, 10 / 21)):
        y = (r["quali_position"] <= thr).astype(float)
        cnt = r.groupby("rank").size().reindex(range(1, 23), fill_value=0)
        hit = y.groupby(r["rank"]).sum().reindex(range(1, 23), fill_value=0)
        out[ev] = ((hit + alpha * prior * 0.2) / (cnt + alpha)).to_numpy()
    return out


def quali_walk_forward(hist, season, rounds, method, cfg=PredictorConfig()) -> pd.DataFrame:
    """Per-driver qualifying predictions for each round from strictly earlier information.
    method: "model" or one of QUALI_BASELINES. Columns: round, driver_id, p_pole, p_q3,
    pred_rank (exp_grid for the model, baseline rank otherwise), actual (quali_position)."""
    out = []
    if not cfg.stick_prequali:
        cfg = replace(cfg, stick_gamma=0.0)  # pre-quali runs without stickiness (T-040)
    stores = _stores_for(hist, cfg, season) if method == "model" else None
    for rnd in rounds:
        truth = _quali_truth(hist, season, rnd)
        if len(truth) < 10:
            continue
        if method == "model":
            p = predict_race(hist, season, rnd, "prequali", cfg, stores=stores)
            d = truth.merge(p[["driver_id", "p_pole", "p_q3", "exp_grid"]], on="driver_id", how="left")
            d = d.rename(columns={"exp_grid": "pred_rank"})
        else:
            d = truth.copy()
            n = len(d)
            if method == "uniform":
                d["p_pole"], d["p_q3"], d["pred_rank"] = 1 / n, min(1.0, 10 / n), np.nan
            else:
                d["pred_rank"] = _quali_baseline_rank(hist, season, rnd, method, d["driver_id"])
                tab = _quali_rank_table(hist, season, rnd, method)
                idx = np.minimum(d["pred_rank"].to_numpy().astype(int), 22) - 1
                pp, p3 = tab["pole"][idx], tab["q3"][idx]
                d["p_pole"] = pp / pp.sum()
                d["p_q3"] = np.minimum(p3 / p3.sum() * 10, 1.0)
        d["round"] = rnd
        out.append(d)
    df = pd.concat(out, ignore_index=True)
    for c in ("p_pole", "p_q3"):
        df[c] = np.clip(df[c], PROB_CLIP, 1 - PROB_CLIP)
    return df


def quali_score(df: pd.DataFrame) -> dict:
    """Mean per-race binary log-loss for pole and Q3, pole NLL (-log P(actual pole sitter)) and
    mean per-race Spearman(predicted rank, actual quali position) (nan for uniform)."""
    out = {}
    for ev, thr in (("pole", 1), ("q3", 10)):
        y = (df["quali_position"] <= thr).astype(float)
        p = df[f"p_{ev}"]
        ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        out[f"{ev}_logloss"] = float(ll.groupby(df["round"]).mean().mean())
    w = df[df["quali_position"] == 1]
    out["pole_nll"] = float(-np.log(w["p_pole"]).mean())
    sp = [g["pred_rank"].corr(g["quali_position"], method="spearman") for _, g in df.groupby("round")]
    out["spearman"] = float(np.nanmean(sp)) if not np.all(np.isnan(sp)) else float("nan")
    out["n_races"] = int(df["round"].nunique())
    return out


def tune_on_season(hist, season, rounds, grid_of_cfgs) -> pd.DataFrame:
    """Walk-forward score for each candidate config on one season (used on
    2025 to pick level-1 hyperparameters without touching 2026)."""
    rows = []
    for cfg in grid_of_cfgs:
        for mode in ("grid", "prequali"):
            s = score(walk_forward(hist, season, rounds, f"model_{mode}", cfg))
            rows.append({"halflife": cfg.halflife_races, "lambda_team": cfg.lambda_team,
                         "lambda_driver": cfg.lambda_driver, "mode": mode, **s})
    return pd.DataFrame(rows)


def by_race_table(hist: dict, season: int, rounds, cfg: PredictorConfig = PredictorConfig(),
                  mode: str = "grid", driver_rows: list | None = None) -> pd.DataFrame:
    """T-014 (T-044: if `driver_rows` is a list, per-driver prediction + actual result rows are
    appended to it for the race-picker API): one row per race of a strictly walk-forward prediction (each
    race predicted from earlier rounds + its own grid) vs the real result:
    predicted/actual winner and podium, P(actual winner), winner NLL,
    per-race binary log-losses, Spearman(grid, finish) and the track
    stickiness numbers used (multiplier is 1.0 when cfg.stick_gamma == 0)."""
    res = hist["results"]
    rho = _rho_table(hist)
    stores = _stores_for(hist, cfg, season)
    rows = []
    for rnd in rounds:
        p = predict_race(hist, season, rnd, mode, cfg, stores=stores)
        t = _attach_truth(p[["driver_id", "code", "p_win", "p_podium", "p_top10", "exp_pos", "exp_pts", "pred_rank"]], hist, season, rnd)
        for ev in ("win", "podium", "top10"):
            t[f"p_{ev}"] = np.clip(t[f"p_{ev}"], PROB_CLIP, 1 - PROB_CLIP)
        r = res[(res["season"] == season) & (res["round"] == rnd)]
        code_of = dict(zip(r["driver_id"], r["code"]))
        if driver_rows is not None:
            act = r.set_index("driver_id")
            for _, d in p.iterrows():
                a = act.loc[d["driver_id"]] if d["driver_id"] in act.index else None
                driver_rows.append({
                    "round": rnd, "driver_id": d["driver_id"], "code": d["code"], "team": d["team"],
                    "grid": d["grid"], "p_win": d["p_win"], "p_podium": d["p_podium"], "p_top10": d["p_top10"],
                    "exp_pos": d["exp_pos"], "exp_pts": d["exp_pts"], "pred_rank": d["pred_rank"], "p_dnf": d["p_dnf"],
                    "actual_position": None if a is None else a["position"],
                    "actual_classified": None if a is None else bool(a["classified"]),
                    "actual_status": None if a is None else a["status"],
                })
        win = t[t["y_win"] == 1]
        pod_act = t[t["y_podium"] == 1].sort_values("position")
        pod_pred = t.sort_values("pred_rank").head(3)  # T-045: top 3 by expected points (T-042 used expected finish)
        ll = lambda ev: float(np.mean(-(t[f"y_{ev}"] * np.log(t[f"p_{ev}"]) + (1 - t[f"y_{ev}"]) * np.log(1 - t[f"p_{ev}"]))))
        st = track_stickiness(hist, season, rnd, cfg)
        rr = rho[(rho["season"] == season) & (rho["round"] == rnd)]
        rows.append({
            "round": rnd, "race": r["race_name"].iloc[0], "circuit_id": r["circuit_id"].iloc[0],
            "pred_winner": p.iloc[0]["code"], "pred_winner_p": float(p.iloc[0]["p_win"]),
            "actual_winner": win["code"].iloc[0] if len(win) else None,
            "actual_winner_grid": int(r[r["position"] == 1]["grid"].iloc[0]) if len(win) else None,
            "p_actual_winner": float(win["p_win"].iloc[0]) if len(win) else np.nan,
            "winner_nll": float(-np.log(win["p_win"].iloc[0])) if len(win) else np.nan,
            "pred_podium": " ".join(pod_pred["code"]), "actual_podium": " ".join(pod_act["code"]),
            "podium_hits": len(set(pod_pred["code"]) & set(pod_act["code"])),
            "win_ll": ll("win"), "podium_ll": ll("podium"), "top10_ll": ll("top10"),
            "spearman_grid_result": float(rr["rho"].iloc[0]) if len(rr) else np.nan,
            "stick_circuit": st["s_circuit"], "stick_global": st["s_global"], "stick_mult": st["mult"],
            "stick_n_current_season": st["n_cur"], "stick_n_earlier_seasons": st["n_old"],
        })
    return pd.DataFrame(rows)
