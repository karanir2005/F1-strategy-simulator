"""
Jolpica-F1 (Ergast-compatible) client with an on-disk JSON cache, plus
flatteners that turn the nested MRData responses into tidy DataFrames for
the race-outcome predictor (predictor.py, T-011).

Why Jolpica and not FastF1 (2026-10-19): since ~2026-10-15 F1's live-timing
endpoints return 401 for every season (see the DATA ACCESS INCIDENT section
in team/BOARD.md), so FastF1 can't load any session that isn't already in
cache/. Jolpica is free and has everything the predictor needs: race
results (incl. starting grid *with penalties*, status, points), qualifying
classification, sprint results and the season schedule. It has no tyre,
track-status or weather data, which the predictor doesn't use.

Rate limits (Jolpica docs: burst 4 req/s, sustained 500 req/h): every
network request sleeps MIN_REQUEST_INTERVAL_S first, and every non-empty
response is cached under data/jolpica/raw/ so a re-run makes no requests
for data it already has. A full cold pull (2022-2026) is ~120 requests.

Caching rule: a response is written to disk only if it contains at least
one race row. Empty responses (a session that hasn't happened yet, or
whose results Jolpica hasn't published yet) are never cached, so the
Saturday rerun (T-013) automatically picks up the US GP qualifying as soon
as Jolpica has it. The season schedule is the one mutable endpoint; it is
refetched when older than SCHEDULE_TTL_S. Per-round results/qualifying for
past rounds are treated as immutable once published (post-race penalty
reclassifications are rare; delete the matching file under
data/jolpica/raw/ to force a refetch).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.jolpi.ca/ergast/f1"
REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = REPO_ROOT / "data" / "jolpica" / "raw"

MIN_REQUEST_INTERVAL_S = 1.0  # well under the 4 req/s burst limit; ~3600/h worst
# case would breach 500/h, but a cold pull is ~120 requests and is cached after.
SCHEDULE_TTL_S = 6 * 3600
PAGE_LIMIT = 100  # Jolpica's max page size
REQUEST_TIMEOUT_S = 30
MAX_RETRIES = 4

_last_request_t = 0.0


def _cache_path(path: str, offset: int) -> Path:
    safe = path.strip("/").replace("/", "_")
    return RAW_DIR / f"{safe}__o{offset}.json"


def _http_get(path: str, offset: int) -> dict:
    global _last_request_t
    url = f"{BASE_URL}/{path.strip('/')}.json"
    params = {"limit": PAGE_LIMIT, "offset": offset}
    for attempt in range(MAX_RETRIES):
        wait = MIN_REQUEST_INTERVAL_S - (time.monotonic() - _last_request_t)
        if wait > 0:
            time.sleep(wait)
        _last_request_t = time.monotonic()
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_S)
        if resp.status_code == 429:
            # rate limited: back off hard rather than hammering
            time.sleep(10 * (attempt + 1))
            continue
        if resp.status_code >= 500:
            time.sleep(3 * (attempt + 1))
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError(f"Jolpica request failed after {MAX_RETRIES} tries: {url}")


def _n_rows(payload: dict) -> int:
    """Count leaf rows on one page (Results/QualifyingResults/SprintResults
    are nested inside Races; the schedule's rows are the Races themselves)."""
    races = payload["MRData"].get("RaceTable", {}).get("Races", [])
    n = 0
    for race in races:
        leaf = None
        for key in ("Results", "QualifyingResults", "SprintResults"):
            if key in race:
                leaf = race[key]
                break
        n += len(leaf) if leaf is not None else 1
    return n


def get_pages(path: str, ttl_s: float | None = None, offline: bool = False) -> list[dict]:
    """
    Fetch every page of a Jolpica endpoint (e.g. "2025/results",
    "2026/18/qualifying"), using the disk cache. Returns the list of raw
    MRData payloads. `ttl_s=None` means a cached non-empty page never
    expires. `offline=True` never touches the network (cache or nothing).
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    pages = []
    offset = 0
    while True:
        cp = _cache_path(path, offset)
        payload = None
        if cp.exists():
            fresh = ttl_s is None or (time.time() - cp.stat().st_mtime) < ttl_s
            if fresh or offline:
                payload = json.loads(cp.read_text(encoding="utf-8"))
        if payload is None:
            if offline:
                break
            payload = _http_get(path, offset)
            if _n_rows(payload) > 0:
                cp.write_text(json.dumps(payload), encoding="utf-8")
        pages.append(payload)
        total = int(payload["MRData"]["total"])
        offset += PAGE_LIMIT
        if offset >= total:
            break
    return pages


def _races(pages: list[dict]) -> list[dict]:
    """Merge Races across pages (a race's rows can straddle a page break)."""
    merged: dict[tuple, dict] = {}
    for p in pages:
        for race in p["MRData"].get("RaceTable", {}).get("Races", []):
            key = (int(race["season"]), int(race["round"]))
            if key not in merged:
                merged[key] = {k: v for k, v in race.items()}
                for leaf in ("Results", "QualifyingResults", "SprintResults"):
                    if leaf in race:
                        merged[key][leaf] = list(race[leaf])
            else:
                for leaf in ("Results", "QualifyingResults", "SprintResults"):
                    if leaf in race:
                        merged[key].setdefault(leaf, []).extend(race[leaf])
    return [merged[k] for k in sorted(merged)]


# ---------------------------------------------------------------- flatteners


def schedule(season: int, offline: bool = False) -> pd.DataFrame:
    pages = get_pages(f"{season}", ttl_s=SCHEDULE_TTL_S, offline=offline)
    rows = []
    for r in _races(pages):
        rows.append(
            {
                "season": int(r["season"]),
                "round": int(r["round"]),
                "race_name": r["raceName"],
                "circuit_id": r["Circuit"]["circuitId"],
                "date": r["date"],
                "has_sprint": "Sprint" in r,
                "quali_date": r.get("Qualifying", {}).get("date"),
                "quali_time": r.get("Qualifying", {}).get("time"),
                "race_time": r.get("time"),
            }
        )
    return pd.DataFrame(rows)


def _int_or_none(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _result_rows(races: list[dict], leaf: str, session: str) -> list[dict]:
    rows = []
    for r in races:
        for res in r.get(leaf, []):
            pos_text = res.get("positionText", res.get("position"))
            rows.append(
                {
                    "season": int(r["season"]),
                    "round": int(r["round"]),
                    "race_name": r["raceName"],
                    "circuit_id": r["Circuit"]["circuitId"],
                    "date": r["date"],
                    "session": session,
                    "driver_id": res["Driver"]["driverId"],
                    "code": res["Driver"].get("code", res["Driver"]["driverId"][:3].upper()),
                    "team_id": res["Constructor"]["constructorId"],
                    "team": res["Constructor"]["name"],
                    "position": _int_or_none(res.get("position")),
                    "position_text": pos_text,
                    # classified = has a numeric positionText (R/D/W/N/E/F are not)
                    "classified": str(pos_text).isdigit(),
                    "grid": _int_or_none(res.get("grid")),
                    "status": res.get("status"),
                    "points": float(res.get("points", 0) or 0),
                    "laps": _int_or_none(res.get("laps")),
                }
            )
    return rows


def results(season: int, round_: int | None = None, offline: bool = False) -> pd.DataFrame:
    path = f"{season}/{round_}/results" if round_ else f"{season}/results"
    return pd.DataFrame(_result_rows(_races(get_pages(path, offline=offline)), "Results", "race"))


def sprint_results(season: int, round_: int | None = None, offline: bool = False) -> pd.DataFrame:
    path = f"{season}/{round_}/sprint" if round_ else f"{season}/sprint"
    return pd.DataFrame(
        _result_rows(_races(get_pages(path, offline=offline)), "SprintResults", "sprint")
    )


QUALI_PENDING_TTL_S = 600


def qualifying(season: int, round_: int | None = None, offline: bool = False) -> pd.DataFrame:
    path = f"{season}/{round_}/qualifying" if round_ else f"{season}/qualifying"
    # Until the round has race results, the qualifying page may be partial or later corrected
    # (DSQ, reclassification): refetch on a short TTL instead of freezing the first response.
    ttl = None
    if round_ and not _cache_path(f"{season}/{round_}/results", 0).exists():
        ttl = QUALI_PENDING_TTL_S
    races = _races(get_pages(path, ttl_s=ttl, offline=offline))
    rows = []
    for r in races:
        for res in r.get("QualifyingResults", []):
            rows.append(
                {
                    "season": int(r["season"]),
                    "round": int(r["round"]),
                    "circuit_id": r["Circuit"]["circuitId"],
                    "driver_id": res["Driver"]["driverId"],
                    "code": res["Driver"].get("code", res["Driver"]["driverId"][:3].upper()),
                    "team_id": res["Constructor"]["constructorId"],
                    "team": res["Constructor"]["name"],
                    "quali_position": _int_or_none(res.get("position")),
                    "q1": res.get("Q1"),
                    "q2": res.get("Q2"),
                    "q3": res.get("Q3"),
                }
            )
    return pd.DataFrame(rows)


def season_tables(season: int, offline: bool = False, per_round_upto: int | None = None):
    """
    (results, qualifying, sprint) DataFrames for a whole season.

    Completed seasons use the paginated season endpoints. For the season in
    progress pass `per_round_upto` = the last round to load: per-round
    endpoints are immutable once published so each is cached separately
    and a new round never invalidates older ones.
    """
    if per_round_upto is None:
        return (
            results(season, offline=offline),
            qualifying(season, offline=offline),
            sprint_results(season, offline=offline),
        )
    sched = schedule(season, offline=offline)
    res, qual, spr = [], [], []
    for _, ev in sched[sched["round"] <= per_round_upto].iterrows():
        res.append(results(season, ev["round"], offline=offline))
        qual.append(qualifying(season, ev["round"], offline=offline))
        if ev["has_sprint"]:
            spr.append(sprint_results(season, ev["round"], offline=offline))
    cat = lambda xs: pd.concat([x for x in xs if len(x)], ignore_index=True) if any(len(x) for x in xs) else pd.DataFrame()
    return cat(res), cat(qual), cat(spr)


DERIVED_DIR = REPO_ROOT / "data" / "jolpica" / "derived"


def load_all(
    completed_seasons=(2022, 2023, 2024, 2025),
    current_season: int = 2026,
    current_upto_round: int = 30,
    offline: bool = False,
    write_csv: bool = True,
):
    """
    Load results / qualifying / sprint for the completed seasons plus the
    current season up to `current_upto_round` (rounds that haven't happened
    simply come back empty). Returns (results, qualifying, sprint, schedule)
    with a `season` column on every row (the era tag ROADMAP asks for), and
    writes them to data/jolpica/derived/*.csv for inspection.
    """
    res, qual, spr = [], [], []
    for s in completed_seasons:
        r, q, sp = season_tables(s, offline=offline)
        res.append(r), qual.append(q), spr.append(sp)
    r, q, sp = season_tables(current_season, offline=offline, per_round_upto=current_upto_round)
    res.append(r), qual.append(q), spr.append(sp)
    scheds = [schedule(s, offline=offline) for s in (*completed_seasons, current_season)]

    def cat(xs):
        xs = [x for x in xs if len(x)]
        return pd.concat(xs, ignore_index=True) if xs else pd.DataFrame()

    out = cat(res), cat(qual), cat(spr), cat(scheds)
    if write_csv:
        DERIVED_DIR.mkdir(parents=True, exist_ok=True)
        for name, df in zip(("results", "qualifying", "sprint", "schedule"), out):
            df.to_csv(DERIVED_DIR / f"{name}.csv", index=False)
    return out
