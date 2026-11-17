"""
MCP server (T-054): exposes the simulator's models and the project notes to
Claude Desktop / Claude Code over stdio, so the user can chat with their own
model on a Claude subscription (no API key; Claude does the generation).

Run:  .venv\\Scripts\\python.exe -m app.mcp_server      (cwd = backend/)

Design choices
- In-process: tools call the existing core/service/route functions directly
  (the FastAPI route functions are plain callables taking pydantic models),
  so no uvicorn is needed and the API and MCP share one code path. HTTPException
  is turned into an {"error": ...} result rather than a protocol error, so the
  model can read the message and retry.
- stdout is the protocol channel on stdio. Anything the data code prints is
  redirected to stderr for the duration of each tool call (_quiet), and
  logging goes to stderr.
- Every result carries a short `caveats` list so Claude does not overstate
  the numbers. These are taken from the module docstrings (monte_carlo.py,
  optimizer.py, tire_model.py), the prediction disclaimer and T-051.
- search_project_notes is the RAG piece: BM25 (pure Python, no embeddings, no
  key) over markdown chunks split by heading. Claude generates the answer
  from the returned snippets.

SDK note: the installed `mcp` is 2.x, where FastMCP was renamed MCPServer
(`from mcp.server.mcpserver import MCPServer`). Pinned in requirements.txt.
"""

from __future__ import annotations

import contextlib
import logging
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
log = logging.getLogger("f1-mcp")

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MAX_LAP_ROWS = 200
MAX_NOTE_HITS = 10

# ----------------------------------------------------------------- caveats

CAVEATS_SIM = [
    "The strategy simulator models a single car in isolation: no traffic, opponents, undercut/overcut or reactive pit calls.",
    "Lap-time and pit-loss noise are additive iid Gaussian; replay errors against real races are tens of seconds, so gaps under ~1 s are not meaningful.",
    "win_probability here means 'beats the other strategies passed in', not a race-win chance; under common random numbers it is mostly 0 or 1. Prefer the delta_* fields.",
    "Safety cars, if enabled, use a fixed pit plan (no reactive stops); default is off.",
    "SOFT/MEDIUM/HARD tie on pace in the 2026 fit (they differ only by wear), and tyre-cliff penalties are still placeholders.",
]
CAVEATS_OPT = [
    "The optimizer mostly finds 1-stop plans; it ranks plans for one car in clean air, not for a race against a field.",
    "Stints are capped at observed typical stint lengths per compound at the track and are at least max(8 laps, 10% of distance); all stints start on new tyres.",
    "Plans within ~1 s of each other are equivalent under this model.",
] + CAVEATS_SIM[1:2] + CAVEATS_SIM[4:]
CAVEATS_PREDICT = [
    "Predictions come from a small Plackett-Luce model refit each race on ~20 races of 2026 data (plus the grid when known); treat probabilities as estimates, not certainties.",
    "Check `mode`: 'pre_quali' simulates the grid from qualifying form, 'grid' uses the real starting grid. Check `stale` too.",
    "No weather, strategy or safety-car modelling in the predictor; DNF is modelled as a separate per-driver probability.",
]
CAVEATS_RESULTS = [
    "Results come from the Jolpica API cache (official classification, including post-race penalties).",
]
CAVEATS_LAPS = [
    f"Rows are capped at {MAX_LAP_ROWS}; narrow the filters if `truncated` is true.",
    "Wet or mixed races (e.g. Brazil 2026, INTERMEDIATE tyres) are in the lap data but excluded from the tyre-model fit.",
    "Per-lap data from OpenF1/FastF1 pulls; SC/VSC/yellow laps are flagged (is_green_flag) and tyre fields can be imputed (tyre_data_imputed).",
]
CAVEATS_NOTES = [
    "Notes are internal working documents and can be out of date; later handoffs override earlier ones. Quote the file path when citing.",
]


def _with_caveats(payload: dict, caveats: list[str]) -> dict:
    payload = dict(payload)
    payload["caveats"] = list(caveats)
    return payload


def _err(msg: str, caveats: list[str]) -> dict:
    return {"error": msg, "caveats": list(caveats)}


@contextlib.contextmanager
def _quiet():
    """Send stray prints (pandas/fastf1/predict script) to stderr, never the protocol stdout."""
    with contextlib.redirect_stdout(sys.stderr):
        yield


def _http_detail(exc: Exception) -> str:
    return str(getattr(exc, "detail", None) or exc)


# ------------------------------------------------------------------- tools


def list_tracks() -> dict:
    """List the circuits the strategy tools know about (track_id, name, total_laps, pit_loss_s, base_laptime_s)."""
    from app.core.track_params import TrackParamsUnavailable, load_tracks

    try:
        tracks = load_tracks()
    except TrackParamsUnavailable as exc:
        return _err(str(exc), CAVEATS_SIM)
    keep = ("track_id", "name", "circuit", "total_laps", "pit_loss_s", "base_laptime_s",
            "on_2026_calendar", "already_raced_2026")
    return _with_caveats(
        {"tracks": [{k: t.get(k) for k in keep} for t in tracks]},
        ["Track parameters (pit loss, base lap time, stint caps) are fitted from pulled races; base lap time is a fresh-tyre low-fuel reference, not a qualifying lap."],
    )


def predict_race(round: int | None = None, top_n: int = 22) -> dict:
    """Win / podium / points probabilities for a race. round=None means the next race (live model, can take a while);
    a past round returns the walk-forward prediction made before it with the actual result alongside."""
    from app.services import prediction_service as svc

    try:
        with _quiet():
            payload = svc.get_next() if round is None else svc.get_past_race(int(round))
    except svc.PredictionUnavailable as exc:
        return _err(f"prediction unavailable: {exc}", CAVEATS_PREDICT)
    except KeyError:
        return _err(f"no precomputed prediction for round {round}; use round=None for the next race", CAVEATS_PREDICT)
    drivers = sorted(payload["drivers"], key=lambda d: (d.get("pred_rank") or 99, -d["p_win"]))[: max(1, top_n)]
    keep = ("code", "name", "team", "grid", "pred_rank", "p_win", "p_podium", "p_top10", "exp_pos",
            "p_dnf", "exp_points", "actual")
    out = {
        "race": payload["race"],
        "mode": payload["mode"],
        "stale": payload.get("stale"),
        "stale_reason": payload.get("stale_reason"),
        "generated_at": payload.get("generated_at"),
        "headline": payload["headline"],
        "drivers": [{k: d[k] for k in keep if k in d} for d in drivers],
        "disclaimer": payload.get("disclaimer"),
        "notes": payload.get("evidence", {}).get("notes", []),
    }
    if payload.get("race_done"):
        out["race_done"] = True
    return _with_caveats(out, CAVEATS_PREDICT)


def get_race_results(round: int, season: int = 2026) -> dict:
    """Official race classification for one round (Jolpica data): position, grid, status, points per driver."""
    from app.core import jolpica_client

    try:
        with _quiet():
            df = jolpica_client.results(int(season), int(round), offline=True)
            if df.empty:
                df = jolpica_client.results(int(season), int(round))
    except Exception as exc:  # network/cache failure: report, don't crash the server
        return _err(f"could not load results: {exc}", CAVEATS_RESULTS)
    if df.empty:
        return _err(f"no results for season {season} round {round} (not raced yet?)", CAVEATS_RESULTS)
    rows = []
    for r in df.sort_values("position").to_dict("records"):
        rows.append({
            "position": None if r["position"] != r["position"] else int(r["position"]),
            "code": r["code"], "team": r["team"],
            "grid": None if r["grid"] != r["grid"] else int(r["grid"]),
            "status": r["status"], "points": float(r["points"]), "laps": None if r["laps"] != r["laps"] else int(r["laps"]),
        })
    return _with_caveats(
        {"season": int(season), "round": int(round), "race_name": str(df["race_name"].iloc[0]), "date": str(df["date"].iloc[0]),
         "results": rows},
        CAVEATS_RESULTS,
    )


def _round(x: Any, nd: int = 2) -> Any:
    return round(x, nd) if isinstance(x, float) else x


def optimize_strategy(
    track_id: str,
    max_stops: int = 3,
    n_sims: int = 2000,
    top_n: int = 5,
    total_laps: int | None = None,
    pit_loss_s: float | None = None,
    base_laptime_s: float | None = None,
    use_track_wear: bool = False,
    safety_cars: bool = False,
    seed: int = 0,
) -> dict:
    """Best pit-stop plans for a track (same inputs as POST /optimize). Times in seconds, pit_laps = last lap of each stint."""
    from fastapi import HTTPException
    from pydantic import ValidationError

    from app.api.routes.tracks import OptimizeRequest, optimize_route

    try:
        req = OptimizeRequest(track_id=track_id, max_stops=max_stops, n_sims=n_sims, top_n=top_n, total_laps=total_laps,
                              pit_loss_s=pit_loss_s, base_laptime_s=base_laptime_s, use_track_wear=use_track_wear,
                              safety_cars=safety_cars, seed=seed)
        with _quiet():
            resp = optimize_route(req)
    except ValidationError as exc:
        return _err(f"invalid input: {exc.errors()[0]['loc']}: {exc.errors()[0]['msg']}", CAVEATS_OPT)
    except HTTPException as exc:
        return _err(_http_detail(exc), CAVEATS_OPT)
    d = resp.model_dump()
    d["track"] = {k: d["track"].get(k) for k in ("track_id", "name", "total_laps", "pit_loss_s", "base_laptime_s")}
    d["strategies"] = [{k: _round(v) for k, v in s.items()} for s in d["strategies"]]
    return _with_caveats(d, CAVEATS_OPT)


def simulate_strategy(
    strategies: list[dict],
    track_id: str | None = None,
    base_laptime: float | None = None,
    n_simulations: int = 2000,
    pit_loss_seconds: float | None = None,
    safety_cars: bool = False,
    seed: int | None = 0,
    include_lap_times: bool = False,
) -> dict:
    """Monte Carlo comparison of explicit pit strategies (same inputs as POST /simulate).
    strategies = [{"name": "1-stop", "stints": [{"compound": "MEDIUM", "laps": 25, "start_age": 1}, ...]}].
    Every strategy must cover the same total laps. Give track_id or base_laptime."""
    from fastapi import HTTPException
    from pydantic import ValidationError

    from app.api.routes.simulation import SimulateRequest, simulate

    try:
        req = SimulateRequest(strategies=strategies, track_id=track_id, base_laptime=base_laptime,
                              n_simulations=n_simulations, pit_loss_seconds=pit_loss_seconds,
                              safety_cars=safety_cars, seed=seed)
        with _quiet():
            resp = simulate(req)
    except ValidationError as exc:
        e = exc.errors()[0]
        return _err(f"invalid input: {'.'.join(map(str, e['loc']))}: {e['msg']}", CAVEATS_SIM)
    except HTTPException as exc:
        return _err(_http_detail(exc), CAVEATS_SIM)
    d = resp.model_dump()
    for r in d["results"]:
        if not include_lap_times:
            r.pop("lap_times", None)
        for k in list(r):
            if isinstance(r[k], float):
                r[k] = round(r[k], 3)
    return _with_caveats(d, CAVEATS_SIM)


def query_laps(
    race: str,
    driver: str | None = None,
    lap_from: int | None = None,
    lap_to: int | None = None,
    compound: str | None = None,
    green_only: bool = False,
    columns: list[str] | None = None,
    max_rows: int = MAX_LAP_ROWS,
) -> dict:
    """Filtered per-lap data from data/processed. `race` is a name fragment, e.g. 'Austin' or 'Brazil' (2026 races).
    Returns at most 200 rows; `truncated` says if more matched."""
    import pandas as pd

    files = sorted(PROCESSED_DIR.glob("*_laps.csv"))
    names = [f.name.replace("_laps.csv", "") for f in files]
    hits = [f for f, n in zip(files, names) if race.strip().lower() in n.lower()]
    if not hits:
        return _err(f"no race matching '{race}'. Available: {', '.join(names)}", CAVEATS_LAPS)
    if len(hits) > 1:
        exact = [f for f in hits if f.name.lower().startswith(race.strip().lower() + "_")]
        if len(exact) != 1:
            return _err(f"'{race}' is ambiguous: {', '.join(f.name.replace('_laps.csv', '') for f in hits)}", CAVEATS_LAPS)
        hits = exact
    df = pd.read_csv(hits[0])
    if driver:
        df = df[df["driver"].str.upper() == driver.strip().upper()]
    if lap_from is not None:
        df = df[df["lap_number"] >= lap_from]
    if lap_to is not None:
        df = df[df["lap_number"] <= lap_to]
    if compound:
        df = df[df["compound"].str.upper() == compound.strip().upper()]
    if green_only:
        df = df[df["is_green_flag"].astype(bool)]
    default_cols = ["driver", "team", "lap_number", "lap_time_seconds", "compound", "tire_age", "position",
                    "pit_in_lap", "pit_out_lap", "is_green_flag", "track_status", "gap_ahead_s"]
    cols = columns or default_cols
    bad = [c for c in cols if c not in df.columns]
    if bad:
        return _err(f"unknown columns {bad}; available: {list(df.columns)}", CAVEATS_LAPS)
    cap = max(1, min(int(max_rows), MAX_LAP_ROWS))
    total = len(df)
    out = df[cols].head(cap)
    rows = [{k: (None if v != v else (_round(v, 3) if isinstance(v, float) else (v.item() if hasattr(v, "item") else v)))
             for k, v in r.items()} for r in out.to_dict("records")]
    return _with_caveats(
        {"race": hits[0].name.replace("_laps.csv", ""), "n_matched": total, "n_returned": len(rows),
         "truncated": total > cap, "rows": rows},
        CAVEATS_LAPS,
    )


# ------------------------------------------------------- BM25 note search

NOTE_GLOBS = ["team/research/*.md", "team/handoffs/*.md", "team/ROADMAP.md", "README.md"]
_STOP = frozenset("a an and are as at be but by for from has have in is it its of on or that the this to was were with".split())
_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP]


def chunk_markdown(text: str, rel_path: str) -> list[dict]:
    """Split on markdown headings (ignoring '#' lines inside code fences). Text before the first heading
    gets the heading '(intro)'. Empty sections are dropped."""
    chunks: list[dict] = []
    heading, buf, fenced = "(intro)", [], False

    def flush():
        body = "\n".join(buf).strip()
        if body:
            chunks.append({"path": rel_path, "heading": heading, "text": body})

    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
        m = None if fenced else _HEADING.match(line)
        if m:
            flush()
            heading, buf = m.group(2), []
        else:
            buf.append(line)
    flush()
    return chunks


class BM25Index:
    """Okapi BM25 over chunks. The heading and file name are indexed with the body so a query that names a
    task id (T-047) or a section title finds the right chunk."""

    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.tf = [Counter(_tokens(f"{c['heading']} {c['heading']} {Path(c['path']).stem} {c['text']}")) for c in chunks]
        self.dl = [sum(t.values()) for t in self.tf]
        self.avgdl = (sum(self.dl) / len(self.dl)) if self.dl else 0.0
        df: Counter = Counter()
        for t in self.tf:
            df.update(t.keys())
        n = len(chunks)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}

    def search(self, query: str, k: int = 5) -> list[tuple[float, dict]]:
        q = _tokens(query)
        scored = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            for w in q:
                f = tf.get(w)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl))
            if s > 0:
                scored.append((s, self.chunks[i]))
        scored.sort(key=lambda x: -x[0])
        return scored[:k]


def build_notes_index(root: Path = REPO_ROOT) -> BM25Index:
    chunks: list[dict] = []
    for pattern in NOTE_GLOBS:
        for p in sorted(root.glob(pattern)):
            try:
                text = p.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            chunks.extend(chunk_markdown(text, p.relative_to(root).as_posix()))
    return BM25Index(chunks)


_notes_index: BM25Index | None = None


def _snippet(text: str, query: str, width: int = 500) -> str:
    """~width chars starting a little before the first query-term hit, on a line boundary where possible."""
    flat = text.strip()
    if len(flat) <= width:
        return flat
    low = flat.lower()
    pos = min((low.find(t) for t in set(_tokens(query)) if low.find(t) >= 0), default=0)
    start = max(0, pos - 80)
    nl = flat.rfind("\n", 0, start)
    start = nl + 1 if nl >= 0 and start - nl < 120 else start
    s = flat[start : start + width].strip()
    return ("..." if start else "") + s + ("..." if start + width < len(flat) else "")


def search_project_notes(query: str, k: int = 5) -> dict:
    """BM25 search over the project's research notes, handoffs, ROADMAP and README (chunked by heading).
    Use it to answer 'why' questions about the models; quote the returned path and heading."""
    global _notes_index
    if _notes_index is None:
        _notes_index = build_notes_index()
    k = max(1, min(int(k), MAX_NOTE_HITS))
    hits = _notes_index.search(query, k)
    return _with_caveats(
        {"query": query, "n_chunks_indexed": len(_notes_index.chunks),
         "hits": [{"path": c["path"], "heading": c["heading"], "score": round(s, 3), "snippet": _snippet(c["text"], query)}
                  for s, c in hits]},
        CAVEATS_NOTES,
    )


CAVEATS_GRID = [
    "Full-grid simulation: pace comes from qualifying and earlier-race ratings; its scaling, noise, pass rate and SC "
    "bunching are fitted on 19 dry 2026 races (T-060; SC bunching and the strategy mix on thin evidence). It ranks "
    "finishers about as well as the grid order / Plackett-Luce predictor, no better (see team/handoffs/T-060-data-scientist.md), "
    "and its win probabilities are more diffuse; not for exact gaps.",
    "Strategies are fixed in advance per car (default: optimizer top plans with per-car variation); no reactive stops, "
    "no lapping/blue flags, red flags, weather or penalties.",
    "Overtake, dirty-air and lap-1 parameters are 2026 fits (T-053); the finishing-position distribution is a model "
    "output, not a forecast guarantee.",
]


def simulate_grid(
    track_id: str,
    round: int | None = None,
    n_sims: int = 1000,
    seed: int = 0,
    safety_cars: bool = True,
    forced_sc_lap: int | None = None,
    forced_sc_laps: int | None = None,
    strategies: dict | None = None,
    top_n: int = 22,
) -> dict:
    """Simulate the whole grid together (same inputs as POST /simulate/grid): per-driver P(win), P(podium),
    expected points, mean finishing position. forced_sc_lap = N forces a safety car starting on lap N (what-if).
    strategies = {"NOR": [{"compound": "MEDIUM", "laps": 25}, {"compound": "HARD", "laps": 31}]} overrides
    those drivers' plans (other cars keep the default). Defaults to the race's real grid / latest qualifying."""
    from fastapi import HTTPException
    from pydantic import ValidationError

    from app.api.routes.grid_sim import GridSimRequest, simulate_grid_route

    try:
        req = GridSimRequest(track_id=track_id, round=round, n_sims=n_sims, seed=seed, safety_cars=safety_cars,
                             forced_sc_lap=forced_sc_lap, forced_sc_laps=forced_sc_laps, strategies=strategies)
        with _quiet():
            resp = simulate_grid_route(req)
    except ValidationError as exc:
        e = exc.errors()[0]
        return _err(f"invalid input: {'.'.join(map(str, e['loc']))}: {e['msg']}", CAVEATS_GRID)
    except HTTPException as exc:
        return _err(_http_detail(exc), CAVEATS_GRID)
    d = resp.model_dump()
    keep = ("code", "team", "grid", "p_win", "p_podium", "p_dnf", "exp_points", "mean_position", "p50_time_s", "strategy")
    rows = []
    for r in d["drivers"][: max(1, int(top_n))]:
        row = {k: r[k] for k in keep}
        row["strategy"] = [f"{x['compound'][0]}{x['laps']}" for x in row["strategy"]]
        rows.append({k: _round(v, 3) for k, v in row.items()})
    out = {k: d[k] for k in ("track_id", "round", "total_laps", "n_sims", "grid_source", "forced_sc_lap",
                             "mean_overtakes_per_race", "mean_sc_laps_per_race")}
    out["mean_overtakes_per_race"] = _round(out["mean_overtakes_per_race"], 1)
    out["mean_sc_laps_per_race"] = _round(out["mean_sc_laps_per_race"], 1)
    out["drivers"] = rows
    return _with_caveats(out, CAVEATS_GRID + ([f"A safety car is forced from lap {forced_sc_lap}."] if forced_sc_lap else []))


CAVEATS_POLICY = [
    "The rule is searched on one random seed and its gain is scored on another, with a 95% interval; if `significant` is "
    "false there is no reliable gain and the rule text says so.",
    "Scored with the full-grid simulation but every other car keeps a fixed plan (see `if_field_follows` for the case "
    "where all cars follow the rule). Safety-car timing is the fitted model (26 SC, 30 VSC events), not a forecast.",
    "Tyre wear and fuel are not reduced under SC/VSC and the tyre-cliff penalties are placeholders, so the value of a "
    "fresh set late in a stint is model-dependent.",
    "Against real 2026 SC/VSC laps the rule agrees with top-10 drivers' actual stops 71% of the time, no better than a "
    "never-box policy (71%), and matches the compound in about a third of cases: a model finding, not what teams do.",
]


def optimize_pit_policy(
    track_id: str,
    driver: str | None = None,
    base_plan: list[dict] | None = None,
    n_sims: int = 2000,
    n_screen: int = 400,
    seed: int = 0,
    field_sensitivity: bool = False,
) -> dict:
    """Best safety-car-reactive pit rule for one driver at a track (same inputs as POST /strategy/policy): 'if an SC or
    VSC starts between laps a and b and tyres are at least k laps old, box for compound C, else keep the plan', with
    the expected points/position gain over the fixed plan and a 95% interval. base_plan = [{"compound": "MEDIUM",
    "laps": 24}, {"compound": "HARD", "laps": 32}] (default: the optimizer's best plan). Takes ~10 s."""
    from fastapi import HTTPException
    from pydantic import ValidationError

    from app.api.routes.pit_policy import PolicyRequest, policy_route

    try:
        req = PolicyRequest(track_id=track_id, driver=driver, base_plan=base_plan, n_sims=n_sims, n_screen=n_screen,
                            seed=seed, field_sensitivity=field_sensitivity)
        with _quiet():
            resp = policy_route(req)
    except ValidationError as exc:
        e = exc.errors()[0]
        return _err(f"invalid input: {'.'.join(map(str, e['loc']))}: {e['msg']}", CAVEATS_POLICY)
    except HTTPException as exc:
        return _err(_http_detail(exc), CAVEATS_POLICY)
    d = resp.model_dump()
    d.pop("assumptions", None)
    d["base_plan"] = [f"{s['compound'][0]}{s['laps']}" for s in d["base_plan"]]

    def rnd(x):
        if isinstance(x, dict):
            return {k: rnd(v) for k, v in x.items()}
        return _round(x, 3) if isinstance(x, float) else x

    return _with_caveats(rnd(d), CAVEATS_POLICY)


# ------------------------------------------------------------------ server

TOOLS = [list_tracks, predict_race, get_race_results, optimize_strategy, simulate_strategy, simulate_grid,
         optimize_pit_policy, query_laps, search_project_notes]

INSTRUCTIONS = (
    "Tools over the user's own F1 strategy simulator and race predictor (2026 season). Always read the `caveats` "
    "list in each result and mention the relevant ones; the strategy engine simulates one car in isolation and is "
    "not validated for real decisions. For 'why' questions call search_project_notes and quote the file path and heading."
)


def build_server():
    from mcp.server.mcpserver import MCPServer

    server = MCPServer("f1-strategy-simulator", instructions=INSTRUCTIONS)
    for fn in TOOLS:
        server.tool()(fn)
    return server


def main() -> None:
    build_server().run("stdio")


if __name__ == "__main__":
    main()
