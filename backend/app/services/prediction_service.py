"""
Prediction service (T-023): wraps the same code path as `predict_race.py`
(auto grid mode when Jolpica has qualifying, else pre-quali) and shapes it
for the API.

Why it imports the root script instead of copying it: `predict_race.py`
owns the entry-list / grid-building logic (including the "drivers with no
quali time start at the back" rule). Re-implementing it here would let the
API and the CLI drift apart. The script is loaded by file path so the API
doesn't depend on the process cwd or sys.path.

Caching: one in-memory entry, valid 10 min, tagged with (round, mode,
qualifying_published). The key is only known after the (disk-cached) data
load, so the TTL is what avoids recomputing the 10k-sim prediction on every
request. On any failure we fall back to the newest saved prediction CSV
with `stale=True`.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
PRED_DIR = REPO_ROOT / "data" / "jolpica" / "predictions"
RAW_DIR = REPO_ROOT / "data" / "jolpica" / "raw"
EVAL_CSV = REPO_ROOT / "data" / "jolpica" / "eval" / "walk_forward_2026.csv"
BY_RACE_CSV = PRED_DIR / "walk_forward_2026_by_race.csv"
QUALI_EVAL_CSV = REPO_ROOT / "data" / "jolpica" / "eval" / "quali_walk_forward.csv"

CACHE_TTL_S = 600
# After qualifying day starts, a cached pre_quali payload is re-checked quickly so the switch to
# grid mode shows up within ~2 min instead of up to 10.
QUALI_DAY_TTL_S = 120
OVERRIDE_DIR = REPO_ROOT / "data" / "jolpica"
SEASON = 2026

DISCLAIMER = (
    "Probabilities from a Plackett-Luce rating model blended with a rank-frequency table, "
    "fitted on 2022-26 Jolpica results. Grid mode uses the qualifying classification, which does not "
    "include grid penalties, unless an operator-supplied post-penalty grid is in use (see grid_source). Small-sample 2026 evidence (16 races): treat as a calibrated "
    "estimate, not a forecast guarantee."
)


class PredictionUnavailable(RuntimeError):
    """No live prediction and no saved CSV to fall back on."""


STALE_TTL_S = 60  # after a failed refresh, serve the stale payload this long before retrying Jolpica
log = logging.getLogger(__name__)
_cache: dict = {"t": 0.0, "key": None, "payload": None}
_stale: dict = {"t": 0.0, "payload": None}
_lock = threading.Lock()
_script = None


def _predict_script():
    global _script
    if _script is None:
        spec = importlib.util.spec_from_file_location("_predict_race_cli", REPO_ROOT / "predict_race.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _script = mod
    return _script


# ------------------------------------------------------------------ helpers


def driver_names(season: int = SEASON) -> dict[str, str]:
    """driver code -> full name, from the cached Jolpica result payloads (no network)."""
    out: dict[str, str] = {}
    for f in sorted(RAW_DIR.glob(f"{season}_*_results__o0.json")):
        try:
            races = json.loads(f.read_text(encoding="utf-8"))["MRData"]["RaceTable"]["Races"]
        except (OSError, KeyError, ValueError):
            continue
        for race in races:
            for r in race.get("Results", []):
                d = r.get("Driver", {})
                if d.get("code"):
                    out[d["code"]] = f"{d.get('givenName', '')} {d.get('familyName', '')}".strip()
    return out


def evidence(mode: str, grid_source: str | None = None) -> dict:
    """Walk-forward 2026 (R2-17) log-loss of the chosen config vs its natural baseline."""
    ev: dict = {"walk_forward": None, "notes": []}
    try:
        t = pd.read_csv(EVAL_CSV)
        labels = t["rounds"].unique().tolist()
        last = max((l for l in labels if l.startswith("R2-")), key=lambda l: int(l[3:]))
        t = t[t["rounds"] == last].set_index("method")
        base = "baseline: grid" if mode == "grid" else "baseline: champ"
        m = t.loc[f"model {mode} (chosen)"]
        ev["walk_forward"] = {
            "races": int(m["n_races"]),
            "win_logloss": round(float(m["win_logloss"]), 4),
            "baseline_win_logloss": round(float(t.loc[base, "win_logloss"]), 4),
            "mode": mode,
        }
    except (OSError, KeyError, ValueError):
        ev["notes"].append("walk-forward evidence file not available")
    ev["notes"] += [
        "Win log-loss is per-driver binary, averaged per race; lower is better.",
        "Baseline is the " + ("starting-grid" if mode == "grid" else "championship-order") + " rank-frequency table.",
    ]
    if mode == "grid":
        if grid_source == "override_file":
            ev["notes"].append("Grid = post-penalty override file supplied by the operator.")
        else:
            ev["notes"].append("Grid = qualifying classification; penalties are not applied.")
    return ev


def prequali_snapshot(rnd: int) -> dict[str, tuple[float, float]]:
    """code -> (p_win, p_podium) from the newest saved pre-quali CSV of this round (T-029).

    Newest by the YYYYMMDD_HHMM stamp in the filename (not mtime, which a copy can change).
    Empty dict when no such file exists or it is unreadable: the caller then reports null,
    never a made-up number."""
    files = sorted(PRED_DIR.glob(f"{SEASON}_R{int(rnd):02d}_prequali_*.csv"), key=lambda p: p.name)
    for f in reversed(files):
        try:
            t = pd.read_csv(f, usecols=["code", "p_win", "p_podium"])
        except (OSError, ValueError):
            continue
        return {str(c): (float(w), float(p)) for c, w, p in zip(t["code"], t["p_win"], t["p_podium"])}
    return {}


QUALI_COLS = ["p_pole", "p_front_row", "p_q3", "p_q1_exit", "exp_grid"]


def quali_evidence() -> dict | None:
    """2026 walk-forward qualifying evidence (predict_race.py --evaluate-quali): the model vs the
    strongest non-uniform baseline by pole log-loss. None when the eval file is missing."""
    try:
        t = pd.read_csv(QUALI_EVAL_CSV)
        t = t[t["season"] == SEASON].set_index("method")
        m = t.loc["model"]
        b = t.loc[[x for x in t.index if x not in ("model", "uniform")]].sort_values("pole_logloss").iloc[0]
        return {
            "races": int(m["n_races"]),
            "pole_logloss": round(float(m["pole_logloss"]), 4),
            "baseline_pole_logloss": round(float(b["pole_logloss"]), 4),
            "baseline_name": {"prev_quali": "previous-race qualifying order", "champ": "championship order"}.get(b.name, b.name),
            "spearman": round(float(m["spearman"]), 4),
            "baseline_spearman": round(float(b["spearman"]), 4),
            "q3_logloss": round(float(m["q3_logloss"]), 4),
            "baseline_q3_logloss": round(float(b["q3_logloss"]), 4),
        }
    except (OSError, KeyError, ValueError, IndexError):
        return None


def prequali_quali_table(rnd: int) -> pd.DataFrame | None:
    """Newest saved pre-quali CSV of this round that carries the T-035 qualifying columns (older
    files from before T-035 do not, and are skipped). None when there is none."""
    files = sorted(PRED_DIR.glob(f"{SEASON}_R{int(rnd):02d}_prequali_*.csv"), key=lambda p: p.name)
    for f in reversed(files):
        try:
            t = pd.read_csv(f, usecols=["code"] + QUALI_COLS)
        except (OSError, ValueError):
            continue
        return t
    return None


def quali_block(mode: str, t: pd.DataFrame, rnd: int | None, actual: dict[str, int | None]) -> dict | None:
    """The optional `qualifying` response block (T-035). Pre-quali: the live/stale table's own
    simulated qualifying distribution, actual_grid null. Grid mode: the saved pre-quali numbers
    with actual_grid filled from the grid in use. None when no qualifying numbers exist."""
    if mode == "grid":
        q = prequali_quali_table(rnd) if rnd is not None else None
    else:
        q = t if all(c in t.columns for c in QUALI_COLS) else None
    if q is None or q.empty:
        return None
    q = q.sort_values(["p_pole", "p_front_row"], ascending=False)
    drivers = [
        {
            "code": r["code"],
            "p_pole": float(r["p_pole"]),
            "p_front_row": float(r["p_front_row"]),
            "p_q3": float(r["p_q3"]),
            "p_q1_exit": float(r["p_q1_exit"]),
            "exp_grid": float(r["exp_grid"]),
            "actual_grid": actual.get(r["code"]) if mode == "grid" else None,
        }
        for _, r in q.iterrows()
    ]
    return {"available": True, "drivers": drivers, "evidence": quali_evidence()}


def shape(t: pd.DataFrame, mode: str, race: dict, generated_at: str, stale: bool, grid_source: str | None = None) -> dict:
    names = driver_names()
    pq = prequali_snapshot(race.get("round")) if mode == "grid" and race.get("round") is not None else {}
    gcol = t["grid"] if "grid" in t.columns else pd.Series([None] * len(t))
    drivers = []
    for i, r in t.reset_index(drop=True).iterrows():
        g = gcol.iloc[i] if mode == "grid" else None
        drivers.append(
            {
                "code": r["code"],
                "name": names.get(r["code"], str(r.get("driver_id", r["code"])).replace("_", " ").title()),
                "team": r["team"],
                "grid": None if g is None or pd.isna(g) else int(g),
                "p_win": float(r["p_win"]),
                "p_podium": float(r["p_podium"]),
                "p_top10": float(r["p_top10"]),
                "exp_pos": float(r["exp_pos"]),
                "p_dnf": float(r["p_dnf"]),
            }
        )
        if mode == "grid":  # optional fields, null when no pre-quali snapshot exists for the driver
            old = pq.get(r["code"])
            drivers[-1]["p_win_prequali"] = None if old is None else old[0]
            drivers[-1]["p_podium_prequali"] = None if old is None else old[1]
    drivers.sort(key=lambda d: (-d["p_win"], -d["p_podium"]))
    actual = {d["code"]: d["grid"] for d in drivers} if mode == "grid" else {}
    return {
        "qualifying": quali_block(mode, t, race.get("round"), actual),
        "race": race,
        "mode": mode,
        "generated_at": generated_at,
        "stale": stale,
        "grid_source": grid_source if mode == "grid" else None,
        "drivers": drivers,
        "headline": {"winner": drivers[0]["code"], "podium": [d["code"] for d in sorted(drivers, key=lambda d: (d["exp_pos"], -d["p_podium"]))[:3]]},
        "evidence": evidence(mode, grid_source),
        "disclaimer": DISCLAIMER,
    }


def _iso_utc(date, time_) -> str | None:
    """Jolpica date ('2026-10-25') + time ('19:00:00Z') -> '2026-10-25T19:00:00Z'; None if either is missing."""
    if date is None or time_ is None or pd.isna(date) or pd.isna(time_):
        return None
    t = str(time_)
    return f"{date}T{t if t.endswith('Z') else t + 'Z'}"


def _start_times(ev) -> dict:
    g = lambda k: ev[k] if k in ev.index else None
    return {
        "race_start_utc": _iso_utc(g("date"), g("race_time")),
        "quali_start_utc": _iso_utc(g("quali_date"), g("quali_time")),
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------ live path


def compute_live() -> tuple[tuple, dict]:
    """Run the prediction as the CLI does. Returns (cache_key, payload). May raise (network, SystemExit)."""
    S = _predict_script()
    P = S.P
    rnd = S.next_round(False)
    hist, _, sched = S.load(False, rnd, with_fp=S.CONFIG.use_fp)
    ev = sched[(sched["season"] == SEASON) & (sched["round"] == rnd)].iloc[0]
    q = hist["qualifying"]
    # Same decision function as the CLI: an existing data/jolpica/grid_override_2026_RNN.csv
    # wins over the qualifying classification (T-027).
    mode, ent, grid, _src, grid_source, has_q = S.resolve(hist, rnd, "auto", None, override_dir=OVERRIDE_DIR)
    t = P.predict_race(hist, SEASON, rnd, mode, S.CONFIG, entry_list=ent, grid=grid)
    api_mode = "grid" if mode == "grid" else "pre_quali"
    race = {
        "round": int(rnd),
        "name": str(ev["race_name"]).replace("Grand Prix", "GP"),
        "circuit": _circuit_name(str(ev["circuit_id"])),
        "date": str(ev["date"]),
        "quali_date": None if pd.isna(ev["quali_date"]) else str(ev["quali_date"]),
        **_start_times(ev),
    }
    payload = shape(t, mode, race, _now(), False, grid_source)
    payload["mode"] = api_mode
    return (rnd, api_mode, bool(has_q), grid_source), payload


def _circuit_name(circuit_id: str) -> str:
    """Circuit display name from the cached schedule payload; falls back to the id."""
    try:
        races = json.loads((RAW_DIR / f"{SEASON}__o0.json").read_text(encoding="utf-8"))["MRData"]["RaceTable"]["Races"]
        for r in races:
            if r["Circuit"]["circuitId"] == circuit_id:
                return r["Circuit"]["circuitName"]
    except (OSError, KeyError, ValueError):
        pass
    return circuit_id


# ------------------------------------------------------------------ fallback


def latest_saved() -> dict:
    """Newest saved prediction CSV (by filename timestamp), shaped as a stale response."""
    files = sorted(PRED_DIR.glob(f"{SEASON}_R*_*.csv"), key=lambda p: p.name)
    if not files:
        raise PredictionUnavailable("no live prediction available and no saved prediction CSV found")
    f = files[-1]
    t = pd.read_csv(f)
    rnd = int(t["round"].iloc[0])
    mode = str(t["mode"].iloc[0])
    race = {"round": rnd, "name": f"Round {rnd}", "circuit": "", "date": "", "quali_date": None}
    try:
        sched = pd.read_csv(REPO_ROOT / "data" / "jolpica" / "derived" / "schedule.csv")
        ev = sched[(sched["season"] == SEASON) & (sched["round"] == rnd)].iloc[0]
        race = {
            "round": rnd,
            "name": str(ev["race_name"]).replace("Grand Prix", "GP"),
            "circuit": _circuit_name(str(ev["circuit_id"])),
            "date": str(ev["date"]),
            "quali_date": None if pd.isna(ev["quali_date"]) else str(ev["quali_date"]),
            **_start_times(ev),
        }
    except (OSError, IndexError, KeyError, ValueError):
        pass
    gs = t["grid_source"].iloc[0] if "grid_source" in t.columns and pd.notna(t["grid_source"].iloc[0]) else None
    payload = shape(t, mode, race, datetime.fromtimestamp(f.stat().st_mtime, timezone.utc).isoformat(timespec="seconds"), True, gs)
    payload["mode"] = "grid" if mode == "grid" else "pre_quali"
    return payload


def _ttl(payload: dict) -> float:
    qd = payload["race"].get("quali_date")
    if payload["mode"] == "pre_quali" and qd and datetime.now(timezone.utc).date().isoformat() >= qd:
        return QUALI_DAY_TTL_S
    return CACHE_TTL_S


def get_next(now: float | None = None) -> dict:
    """Cached live prediction, else stale fallback, else PredictionUnavailable."""
    now = time.monotonic() if now is None else now
    with _lock:
        if _cache["payload"] is not None and now - _cache["t"] < _ttl(_cache["payload"]):
            return _cache["payload"]
        if _stale["payload"] is not None and now - _stale["t"] < STALE_TTL_S:
            return _stale["payload"]
        try:
            key, payload = compute_live()
        except (Exception, SystemExit) as e:  # network down, Jolpica error, grid mode without quali, ...
            reason = f"{type(e).__name__}: {e}"[:500]
            log.warning("live prediction failed, serving stale CSV: %s", reason, exc_info=isinstance(e, Exception))
            payload = latest_saved()
            payload["stale_reason"] = reason
            _stale.update(t=now, payload=payload)
            return payload
        _stale.update(t=0.0, payload=None)
        _cache.update(t=now, key=key, payload=payload)
        return payload


def clear_cache() -> None:
    _cache.update(t=0.0, key=None, payload=None)
    _stale.update(t=0.0, payload=None)


# ------------------------------------------------------------------ history


def _pole_sitters() -> dict[int, str]:
    """round -> code of the driver who started 1st (Jolpica race-result grid column)."""
    try:
        r = pd.read_csv(REPO_ROOT / "data" / "jolpica" / "derived" / "results.csv", usecols=["season", "round", "code", "grid"])
    except (OSError, ValueError):
        return {}
    r = r[(r["season"] == SEASON) & (r["grid"] == 1)]
    return {int(a): b for a, b in zip(r["round"], r["code"])}


def history() -> list[dict]:
    if not BY_RACE_CSV.exists():
        raise PredictionUnavailable("walk-forward by-race table not found")
    t = pd.read_csv(BY_RACE_CSV)
    poles = _pole_sitters()
    out = []
    for r in t.to_dict("records"):
        pred = str(r["pred_podium"]).split()
        act = str(r["actual_podium"]).split()
        out.append(
            {
                "round": int(r["round"]),
                "race": r["race"],
                "pole_sitter": poles.get(int(r["round"])),
                "predicted_winner": r["pred_winner"],
                "predicted_winner_p": float(r["pred_winner_p"]),
                "actual_winner": r["actual_winner"],
                "actual_winner_grid": None if pd.isna(r["actual_winner_grid"]) else int(r["actual_winner_grid"]),
                "p_actual_winner": float(r["p_actual_winner"]),
                "winner_correct": r["pred_winner"] == r["actual_winner"],
                "predicted_podium": pred,
                "actual_podium": act,
                "podium_hits": int(r["podium_hits"]),
            }
        )
    return out


# ------------------------------------------------------------------ past races (T-044)

DRIVERS_CSV = PRED_DIR / "walk_forward_2026_drivers.csv"
SCHEDULE_CSV = REPO_ROOT / "data" / "jolpica" / "derived" / "schedule.csv"
_races_cache: dict = {"mtime": None, "df": None}


def _drivers_table() -> pd.DataFrame | None:
    """Per-driver walk-forward predictions + actual results (written by predict_race.py --by-race).
    Read once per file mtime; no simulation happens at request time."""
    try:
        m = DRIVERS_CSV.stat().st_mtime
    except OSError:
        return None
    if _races_cache["mtime"] != (str(DRIVERS_CSV), m):
        _races_cache.update(mtime=(str(DRIVERS_CSV), m), df=pd.read_csv(DRIVERS_CSV))
    return _races_cache["df"]


def _schedule() -> pd.DataFrame:
    try:
        s = pd.read_csv(SCHEDULE_CSV)
        return s[s["season"] == SEASON]
    except (OSError, ValueError):
        return pd.DataFrame(columns=["round", "race_name", "date", "circuit_id"])


def list_races() -> list[dict]:
    """Rounds with a precomputed walk-forward prediction, plus the next unraced round (served live
    by /predict/next)."""
    t = _drivers_table()
    done = set() if t is None else {int(r) for r in t["round"].unique()}
    out = []
    for r in _schedule().to_dict("records"):
        rnd = int(r["round"])
        if rnd in done or rnd == max(done, default=0) + 1:
            out.append({"round": rnd, "name": str(r["race_name"]).replace("Grand Prix", "GP"),
                        "date": str(r["date"]), "has_prediction": True, "is_next": rnd not in done})
    return out


def get_past_race(rnd: int) -> dict:
    """Same shape as /predict/next (mode grid, stale false) plus `race_done` and per-driver `actual`.
    Raises KeyError when the round has no precomputed prediction."""
    t = _drivers_table()
    if t is None or rnd not in set(t["round"]):
        raise KeyError(rnd)
    sub = t[t["round"] == rnd]
    sch = _schedule()
    ev = sch[sch["round"] == rnd].iloc[0]
    race = {"round": rnd, "name": str(ev["race_name"]).replace("Grand Prix", "GP"),
            "circuit": _circuit_name(str(ev["circuit_id"])), "date": str(ev["date"]),
            "quali_date": None if pd.isna(ev.get("quali_date")) else str(ev["quali_date"]), **_start_times(ev)}
    generated = datetime.fromtimestamp(DRIVERS_CSV.stat().st_mtime, tz=timezone.utc).isoformat()
    payload = shape(sub, "grid", race, generated, False, None)
    payload["qualifying"] = None
    payload["race_done"] = True
    payload["evidence"]["notes"].append(
        "Walk-forward prediction: made from earlier rounds plus this race's own starting grid (no leakage)."
    )
    act = {r["code"]: r for r in sub.to_dict("records")}
    for d in payload["drivers"]:
        a = act[d["code"]]
        pos = a["actual_position"]
        status = None if pd.isna(a["actual_status"]) else str(a["actual_status"])
        d["actual"] = {
            "position": None if pd.isna(pos) else int(pos),
            "classified": None if pd.isna(a["actual_classified"]) else bool(a["actual_classified"]),
            "status": status,
            "dnf": status is not None and not bool(a["actual_classified"]) and not status.startswith("+") and status != "Finished",
        }
    return payload
