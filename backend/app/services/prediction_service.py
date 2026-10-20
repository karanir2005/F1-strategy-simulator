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

CACHE_TTL_S = 600
SEASON = 2026

DISCLAIMER = (
    "Probabilities from a Plackett-Luce rating model blended with a rank-frequency table, "
    "fitted on 2022-26 Jolpica results. Grid mode uses the qualifying classification and does not "
    "include grid penalties. Small-sample 2026 evidence (16 races): treat as a calibrated "
    "estimate, not a forecast guarantee."
)


class PredictionUnavailable(RuntimeError):
    """No live prediction and no saved CSV to fall back on."""


_cache: dict = {"t": 0.0, "key": None, "payload": None}
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


def evidence(mode: str) -> dict:
    """Walk-forward 2026 (R2-17) log-loss of the chosen config vs its natural baseline."""
    ev: dict = {"walk_forward": None, "notes": []}
    try:
        t = pd.read_csv(EVAL_CSV)
        t = t[t["rounds"] == "R2-17"].set_index("method")
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
        ev["notes"].append("Grid = qualifying classification; penalties are not applied.")
    return ev


def shape(t: pd.DataFrame, mode: str, race: dict, generated_at: str, stale: bool) -> dict:
    names = driver_names()
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
    drivers.sort(key=lambda d: (-d["p_win"], -d["p_podium"]))
    return {
        "race": race,
        "mode": mode,
        "generated_at": generated_at,
        "stale": stale,
        "drivers": drivers,
        "headline": {"winner": drivers[0]["code"], "podium": [d["code"] for d in sorted(drivers, key=lambda d: -d["p_podium"])[:3]]},
        "evidence": evidence(mode),
        "disclaimer": DISCLAIMER,
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
    has_q = len(q[(q["season"] == SEASON) & (q["round"] == rnd)]) > 0
    mode = "grid" if has_q else "prequali"
    ent, grid, _src = S._entry_and_grid(hist, rnd, mode, None)
    t = P.predict_race(hist, SEASON, rnd, mode, S.CONFIG, entry_list=ent, grid=grid)
    api_mode = "grid" if mode == "grid" else "pre_quali"
    race = {
        "round": int(rnd),
        "name": str(ev["race_name"]).replace("Grand Prix", "GP"),
        "circuit": _circuit_name(str(ev["circuit_id"])),
        "date": str(ev["date"]),
        "quali_date": None if pd.isna(ev["quali_date"]) else str(ev["quali_date"]),
    }
    payload = shape(t, mode, race, _now(), False)
    payload["mode"] = api_mode
    return (rnd, api_mode, bool(has_q)), payload


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
    files = sorted(PRED_DIR.glob(f"{SEASON}_R*_*.csv"), key=lambda p: p.stat().st_mtime)
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
        }
    except (OSError, IndexError, KeyError, ValueError):
        pass
    payload = shape(t, mode, race, datetime.fromtimestamp(f.stat().st_mtime, timezone.utc).isoformat(timespec="seconds"), True)
    payload["mode"] = "grid" if mode == "grid" else "pre_quali"
    return payload


def get_next(now: float | None = None) -> dict:
    """Cached live prediction, else stale fallback, else PredictionUnavailable."""
    now = time.monotonic() if now is None else now
    with _lock:
        if _cache["payload"] is not None and now - _cache["t"] < CACHE_TTL_S:
            return _cache["payload"]
        try:
            key, payload = compute_live()
        except (Exception, SystemExit):  # network down, Jolpica error, grid mode without quali, ...
            return latest_saved()
        _cache.update(t=now, key=key, payload=payload)
        return payload


def clear_cache() -> None:
    _cache.update(t=0.0, key=None, payload=None)


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
