"""
T-055 validation: what the safety-car pit policy would have done on the REAL 2026 SC/VSC laps vs what the top-10
finishers actually did.

For each pulled 2026 race: find the policy for that track (pit_policy.search_policy, grid sim, hero = pole sitter,
optimizer plan; simulation-only, it never sees the lap data), find the real SC/VSC starts in the lap data, and
for every top-10 finisher still running on the start lap compare
  policy fires?   window and trigger match, tyre age on the start lap >= min_age, enough laps left, and
  driver boxed?   a pit-in on the start lap or the next lap (the first two laps of the neutralisation).
Agreement = share of (driver, event) pairs where the two answers match. Baselines: "never box" (what the
agreement would be if the policy were silent) and the base rate of boxing. The compound check is among pairs
where both boxed. A real decision depends on the planned stop lap, which the data does not give, so a driver
who was due to stop that lap anyway counts as "boxed" (a known bias toward disagreement on policy-silent
cases).

Not held out (T-059 F2): the tyre model and track entry the policy search uses are fitted on all pulled races,
including the one being checked. The policy never sees that race's pit or SC rows, but this is an in-sample check,
not leave-one-race-out.

Run:  .venv\\Scripts\\python.exe validate_pit_policy.py [--n-screen 400] [--n-eval 1000] [--races Austin,Japan]
Output: data/scorecard/pit_policy_validation.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))

from app.api.routes.simulation import get_tire_model  # noqa: E402
from app.core import pit_policy as PP  # noqa: E402
from app.core.track_params import get_track  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "data" / "scorecard" / "pit_policy_validation.json"
NEUTRAL_FRAC = 0.2  # share of drivers flagged on a lap for it to count as neutralised
BOX_LAPS = 2  # a pit-in on the start lap or the one after counts as reacting to the event


def real_events(laps: pd.DataFrame):
    """[(start_lap, 'sc' | 'vsc')] from the per-lap flags. A VSC that turns into an SC gives both events."""
    out = []
    for kind, col in (("sc", "has_sc"), ("vsc", "has_vsc")):
        frac = laps.groupby("lap_number")[col].mean()
        flagged = frac[frac > NEUTRAL_FRAC].index.tolist()
        prev = None
        for lap in flagged:
            if prev is None or lap != prev + 1:
                out.append((int(lap), kind))
            prev = lap
    return sorted(out)


def policy_fires(pol, kind: str, lap: int, age: float, total: int) -> bool:
    if not (pol.window_lo <= lap <= pol.window_hi):
        return False
    if pol.trigger == "sc" and kind != "sc":
        return False
    return age >= pol.min_age and total - lap + 1 >= PP.MIN_LAPS_AFTER and lap >= 2


def validate_race(name: str, tire_model: dict, n_screen: int, n_eval: int, seed: int):
    track_id = name.lower()
    if get_track(track_id) is None:
        return {"race": name, "skipped": "no track params"}
    laps = pd.read_csv(PROCESSED / f"{name}_2026_laps.csv")
    res = pd.read_csv(PROCESSED / f"{name}_2026_results.csv")
    events = real_events(laps)
    top10 = set(res.loc[res["position"] <= 10, "driver"])
    total = int(laps["lap_number"].max())
    ctx = PP.build_context(track_id, tire_model, seed=seed)
    sr = PP.search_policy(ctx, n_screen=n_screen, n_eval=n_eval, seed=seed, field_sensitivity=False)
    pol = sr.best
    rec = {"race": name, "track_id": track_id, "n_events": len(events), "events": [f"{k}@{l}" for l, k in events],
           "policy": None if pol is None else vars(pol) | {}, "gain_pts": None if sr.evaluation is None else sr.evaluation.gain_pts,
           "gain_pts_ci": None if sr.evaluation is None else sr.evaluation.gain_pts_ci, "significant": sr.significant}
    pairs = []
    for lap, kind in events:
        if lap < 2 or lap >= total:
            continue
        for drv in top10:
            d = laps[laps["driver"] == drv].set_index("lap_number")
            if lap not in d.index or d.loc[lap, "tire_age"] != d.loc[lap, "tire_age"]:
                continue
            age = float(d.loc[lap, "tire_age"])
            boxed_laps = [l for l in range(lap, lap + BOX_LAPS) if l in d.index and int(d.loc[l, "pit_in_lap"]) == 1]
            boxed = bool(boxed_laps)
            nxt = None
            if boxed and boxed_laps[0] + 1 in d.index:
                nxt = str(d.loc[boxed_laps[0] + 1, "compound"])
            fires = False if pol is None else policy_fires(pol, kind, lap, age, total)
            pairs.append({"driver": drv, "lap": lap, "kind": kind, "age": age, "fires": fires, "boxed": boxed,
                          "new_compound": nxt, "policy_compound": None if pol is None else pol.compound})
    rec["pairs"] = pairs
    return rec


def summarise(recs):
    pairs = [p for r in recs for p in r.get("pairs", [])]
    if not pairs:
        return {}
    f = np.array([p["fires"] for p in pairs])
    b = np.array([p["boxed"] for p in pairs])
    both = f & b
    comp = [p["new_compound"] == p["policy_compound"] for p in pairs if p["fires"] and p["boxed"] and p["new_compound"]]
    out = {
        "n_pairs": len(pairs),
        "n_races_with_events": sum(1 for r in recs if r.get("n_events")),
        "agreement": float((f == b).mean()),
        "agreement_never_box_baseline": float((~b).mean()),
        "base_rate_boxed": float(b.mean()),
        "policy_fire_rate": float(f.mean()),
        "precision_P_boxed_given_fires": float(b[f].mean()) if f.any() else None,
        "recall_P_fires_given_boxed": float(f[b].mean()) if b.any() else None,
        "n_fires": int(f.sum()), "n_boxed": int(b.sum()), "n_both": int(both.sum()),
        "compound_match_when_both": float(np.mean(comp)) if comp else None,
        "n_compound_pairs": len(comp),
    }
    # restricted to pairs where the policy was applicable (the event is inside a window and old enough tyres exist)
    for kind in ("sc", "vsc"):
        sel = np.array([p["kind"] == kind for p in pairs])
        if sel.any():
            out[f"agreement_{kind}_events"] = float((f[sel] == b[sel]).mean())
            out[f"n_pairs_{kind}"] = int(sel.sum())
            out[f"base_rate_boxed_{kind}"] = float(b[sel].mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-screen", type=int, default=400)
    ap.add_argument("--n-eval", type=int, default=1000)
    ap.add_argument("--races", default="")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    tm = get_tire_model()
    names = sorted(p.name.replace("_2026_laps.csv", "") for p in PROCESSED.glob("*_2026_laps.csv"))
    if a.races:
        names = [n for n in names if n in a.races.split(",")]
    recs = []
    for n in names:
        try:
            r = validate_race(n, tm, a.n_screen, a.n_eval, a.seed)
        except Exception as exc:  # keep going: one race without params must not stop the run
            r = {"race": n, "skipped": f"{type(exc).__name__}: {exc}"}
        recs.append(r)
        print(n, r.get("events"), r.get("policy"), r.get("skipped", ""), flush=True)
    summ = summarise(recs)
    print(json.dumps(summ, indent=1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"scope": "in-sample: tyre model and track entry are fitted on all races incl. the one checked; policy never sees its pit/SC rows (T-059 F2)", "summary": summ, "races": recs}, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
