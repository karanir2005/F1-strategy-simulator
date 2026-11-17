"""
Leave-one-race-out validation of the full-grid simulation (T-034).

For each 2026 round: build the tyre model WITHOUT that race's laps, take the race's real starting grid and
qualifying (Q1 times), pace/DNF inputs from EARLIER rounds only (grid_inputs.py), run the grid sim, rank the
drivers by expected points (the predictor's rule) and compare with the real finishing order:
  - Spearman correlation (predicted rank vs official finishing position, all drivers incl. retirements)
  - top-3 hit rate (share of the real podium inside the predicted top 3) and winner hit
against (a) the grid order and (b) the Plackett-Luce predictor's walk-forward predictions
(data/jolpica/predictions/walk_forward_2026_drivers.csv, grid mode, already walk-forward).

Known leaks (small, listed in the T-034 handoff): track_params.json (base lap time, pit loss, stint caps) was built
from all 2026 races including the held-out one, so the default strategies and lap-time level see the target
track. They do not carry driver or car pace.

    python validate_grid_sim.py [--n-sims 1000] [--rounds 2-20] [--sweep]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))

from app.core import grid_inputs as G  # noqa: E402
from app.core import tire_model as tm  # noqa: E402
from app.core.race_grid_sim import load_interaction_params, rank_by_points, simulate_grid  # noqa: E402
from app.core.safety_cars import load_params as load_sc  # noqa: E402

WF_PATH = ROOT / "data" / "jolpica" / "predictions" / "walk_forward_2026_drivers.csv"
OUT_PATH = ROOT / "data" / "scorecard" / "grid_sim_validation.json"


def round_to_file() -> dict:
    out = {}
    for p in (ROOT / "data" / "processed").glob("*_race_info.json"):
        info = json.loads(p.read_text())
        if info.get("season") == 2026 and info.get("round"):
            out[int(info["round"])] = p.name.replace("_race_info.json", "_laps.csv")
    return out


def score_order(pred_rank: np.ndarray, actual_pos: np.ndarray, classified: np.ndarray) -> dict:
    rho = spearmanr(pred_rank, actual_pos).statistic
    c = classified.astype(bool)
    rho_c = spearmanr(pred_rank[c], actual_pos[c]).statistic if c.sum() > 3 else np.nan
    pred_top3 = set(np.argsort(pred_rank)[:3])
    act_top3 = set(np.argsort(actual_pos)[:3])
    return {
        "spearman": float(rho),
        "spearman_classified": float(rho_c),
        "top3_hits": len(pred_top3 & act_top3) / 3.0,
        "winner": float(np.argmin(pred_rank) == np.argmin(actual_pos)),
    }


def run(rounds, n_sims, race_pace_scale, quali_weight, laps_all, files, hist, verbose=True, overtake_scale=None):
    rows = []
    wf = pd.read_csv(WF_PATH)
    for rnd in rounds:
        res = hist["results"]
        r = res[(res["season"] == 2026) & (res["round"] == rnd)]
        if r.empty:
            continue
        cid = r["circuit_id"].iloc[0]
        track_id = G.CIRCUIT_TO_TRACK.get(cid)
        if track_id is None:
            continue
        f = files.get(rnd)
        train = laps_all[laps_all["race_file"].str.replace("\\", "/").str.endswith(f)] if f else None
        laps_train = laps_all if f is None else laps_all[~laps_all["race_file"].str.replace("\\", "/").str.endswith(f)]
        model = tm.build_tire_model(laps=laps_train)
        gi = G.build_inputs(track_id, model, rnd=rnd, seed=rnd, hist=hist, race_pace_scale=race_pace_scale, quali_weight=quali_weight)
        ip = load_interaction_params()
        sim = simulate_grid(
            gi.cars, gi.track["base_laptime_s"], model, gi.total_laps, n_sims=n_sims, seed=rnd,
            pit_loss_s=gi.track["pit_loss_s"], interaction=ip, safety_cars=load_sc(), track_id=track_id,
        )
        s = sim.summary()
        codes = [x["code"] for x in s]
        pr = rank_by_points([x["exp_points"] for x in s], [x["mean_position"] for x in s])
        er = r.set_index("code").loc[codes]
        actual = er["position"].to_numpy(dtype=float)
        cls = er["classified"].to_numpy()
        grid = np.array([x["grid"] for x in s])
        row = {"round": rnd, "track": track_id, "n": len(codes), "overtakes_per_race": float(sim.n_overtakes.mean())}
        for name, rk in (("sim", pr), ("grid", grid.astype(float))):
            for k, v in score_order(np.asarray(rk, float), actual, cls).items():
                row[f"{name}_{k}"] = v
        w = wf[wf["round"] == rnd].set_index("code")
        if len(w) == len(codes) and set(w.index) == set(codes):
            for k, v in score_order(w.loc[codes, "pred_rank"].to_numpy(float), actual, cls).items():
                row[f"pl_{k}"] = v
        rows.append(row)
        if verbose:
            print(f"R{rnd:>2} {track_id:<12} sim rho {row['sim_spearman']:.2f} grid {row['grid_spearman']:.2f} "
                  f"pl {row.get('pl_spearman', float('nan')):.2f} | top3 {row['sim_top3_hits']:.2f}/{row['grid_top3_hits']:.2f}/"
                  f"{row.get('pl_top3_hits', float('nan')):.2f} | passes {row['overtakes_per_race']:.0f}", flush=True)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> dict:
    out = {"n_races": int(len(df))}
    for m in ("sim", "grid", "pl"):
        c = f"{m}_spearman"
        if c in df:
            d = df[df[c].notna()]
            out[m] = {
                "spearman": float(d[c].mean()),
                "spearman_classified": float(d[f"{m}_spearman_classified"].mean()),
                "top3_hit_rate": float(d[f"{m}_top3_hits"].mean()),
                "winner_hit_rate": float(d[f"{m}_winner"].mean()),
                "n": int(len(d)),
            }
    if "sim_spearman" in df:
        d = df["sim_spearman"] - df["grid_spearman"]
        out["sim_minus_grid_spearman"] = {"mean": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d))), "races_better": int((d > 0).sum())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sims", type=int, default=1000)
    ap.add_argument("--rounds", default="2-20")
    ap.add_argument("--sweep", action="store_true", help="also try other pace scales / quali weights")
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.rounds.split("-"))
    rounds = list(range(lo, hi + 1))
    hist = G.load_history()
    laps_all = tm.load_all_laps()
    files = round_to_file()
    t0 = time.time()
    configs = [("default", G.RACE_PACE_SCALE, G.QUALI_WEIGHT, True)]
    if a.sweep:
        configs += [(f"scale={s},qw={q}", s, q, False) for s, q in [(0.3, 0.5), (1.0, 0.5), (0.6, 1.0), (1.0, 1.0)]]
    report = {}
    for name, scale, qw, verbose in configs:
        print(f"\n== {name} (race_pace_scale={scale}, quali_weight={qw}, n_sims={a.n_sims})")
        df = run(rounds, a.n_sims, scale, qw, laps_all, files, hist, verbose=verbose)
        report[name] = {"summary": summarise(df), "per_race": df.to_dict("records")}
        s = report[name]["summary"]
        for m in ("sim", "grid", "pl"):
            if m in s:
                print(f"  {m:<5} Spearman {s[m]['spearman']:.3f} (classified {s[m]['spearman_classified']:.3f})  "
                      f"top-3 {s[m]['top3_hit_rate']:.3f}  winner {s[m]['winner_hit_rate']:.3f}  n={s[m]['n']}")
        print("  sim - grid Spearman:", s["sim_minus_grid_spearman"])
    print(f"\n{time.time() - t0:.0f}s")
    if not a.no_write:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
