"""
Race-outcome prediction table (T-011): P(win), P(podium), P(top 10) per
driver for the next 2026 race (round 18, United States GP, by default).

Run from project root:

    python predict_race.py                    # auto: grid mode if Jolpica has the
                                              # target's qualifying, else pre-quali
    python predict_race.py --mode prequali    # force pre-qualifying mode
    python predict_race.py --mode grid        # quali order from Jolpica as the grid
    python predict_race.py --mode grid --grid-file grid.csv
                                              # post-penalty grid (columns: code,grid)
    python predict_race.py --evaluate         # walk-forward 2026 vs baselines
    python predict_race.py --offline          # cache only, no network

Data come from the Jolpica-F1 API (FastF1 live timing is 401 since
2026-10-15, see team/BOARD.md) and are cached under data/jolpica/raw/;
tidy tables are written to data/jolpica/derived/ and each prediction table
to data/jolpica/predictions/. Model details: backend/app/core/predictor.py.

IMPORTANT for the Saturday (T-013) rerun: Jolpica's qualifying endpoint is
the qualifying *classification*, not the starting grid. Grid penalties are
only reflected in Jolpica's race results (after the race). If any penalty
is announced, write the official post-penalty grid to a CSV (code,grid;
pit-lane start = 0) and pass it with --grid-file, or save it as
data/jolpica/grid_override_2026_R18.csv: if that file exists, both this CLI
and GET /predict/next use it instead of the qualifying classification
(T-027). --out-dir redirects the saved prediction CSV (rehearsals).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

import numpy as np
import pandas as pd

from app.core import jolpica_client as J
from app.core import openf1_practice as F
from app.core import predictor as P

SEASON = 2026
OUT_DIR = Path(__file__).resolve().parent / "data" / "jolpica"

# Chosen config: level-1 hyperparameters picked on the 2025 season
# walk-forward (independent of the 2026 evaluation), 2025 priors kept only
# where they beat 2026-only in the 2026 walk-forward. See --evaluate and
# team/handoffs/T-011-data-scientist.md.
CONFIG = P.PredictorConfig()

GRID_QUALI = "qualifying_classification"
GRID_OVERRIDE = "override_file"


def override_grid_path(rnd: int, base_dir: Path | None = None) -> Path:
    """Post-penalty grid override for a round (T-027). If it exists it wins over Jolpica's
    qualifying classification, for the CLI and the API alike."""
    return (base_dir or OUT_DIR) / f"grid_override_{SEASON}_R{rnd:02d}.csv"


def headline(t: pd.DataFrame) -> dict:
    """Winner = highest P(win); podium = top 3 by P(podium). Same rule as the API's headline."""
    w = t.sort_values(["p_win", "p_podium"], ascending=False, kind="stable")
    pod = t.sort_values("p_podium", ascending=False, kind="stable")
    return {"winner": str(w["code"].iloc[0]), "podium": [str(c) for c in pod["code"].iloc[:3]]}


def load_fp(sched: pd.DataFrame, offline: bool) -> pd.DataFrame:
    """OpenF1 practice long-run gaps joined to Jolpica rounds by race date."""
    g = F.weekend_long_run_gaps(SEASON, offline=offline)
    s = sched[sched["season"] == SEASON][["season", "round", "date"]]
    g = g.merge(s, left_on="race_date", right_on="date", how="inner")
    out = OUT_DIR.parent / "openf1_predictor"
    out.mkdir(parents=True, exist_ok=True)
    g.to_csv(out / f"fp_long_run_gaps_{SEASON}.csv", index=False)
    return g


def load(offline: bool, target_round: int, with_fp: bool = False):
    res, qual, spr, sched = J.load_all(current_season=SEASON, current_upto_round=target_round, offline=offline)
    fp = load_fp(sched, offline) if with_fp else None
    tr_path = J.DERIVED_DIR / "results_2018_2021.csv"  # track-stickiness history (T-014)
    tr = pd.read_csv(tr_path) if tr_path.exists() else None
    return P.prepare_history(res, qual, sprint=spr, fp=fp, track_results=tr, schedule=sched), spr, sched


def next_round(offline: bool) -> int:
    sched = J.schedule(SEASON, offline=offline)
    for rnd in sched["round"]:
        if J.results(SEASON, int(rnd), offline=offline).empty:
            return int(rnd)
    raise SystemExit("season complete")


def read_grid_file(path) -> pd.DataFrame:
    """Tolerant override-grid reader: ; or , separator, any header case, stray whitespace,
    codes upper-cased, PL/PIT/blank-free grid cells ("PL"/"pit" = pit-lane start = 0).
    Raises SystemExit with the offending rows listed."""
    g = pd.read_csv(path, sep=None, engine="python", dtype=str, skipinitialspace=True)
    g.columns = [str(c).strip().lower() for c in g.columns]
    if "code" not in g.columns or "grid" not in g.columns:
        raise SystemExit(f"grid file needs columns code,grid; found {list(g.columns)}")
    g = g[["code", "grid"]].copy()
    g["code"] = g["code"].fillna("").str.strip().str.upper()
    raw = g["grid"].fillna("").str.strip().str.upper()
    raw = raw.where(~raw.isin(["PL", "PIT", "PITLANE", "PIT LANE"]), "0")
    num = pd.to_numeric(raw, errors="coerce")
    bad = g.loc[num.isna(), "code"].tolist()
    if bad:
        raise SystemExit(f"grid file has blank/non-numeric grid values for: {bad}")
    g["grid"] = num.astype(int)
    return g[g["code"] != ""].reset_index(drop=True)


def _entry_and_grid(hist, rnd, mode, grid_file):
    """Entry list + grid for the target race (grid mode only)."""
    q = hist["qualifying"]
    q = q[(q["season"] == SEASON) & (q["round"] == rnd)]
    base = P.entrants_for(hist, SEASON, rnd)
    if mode == "prequali":
        return base, None, "entry list from " + ("qualifying" if len(q) else "the latest race")
    if grid_file:
        g = read_grid_file(grid_file)
        dup_c = g["code"][g["code"].duplicated()].unique().tolist()
        if dup_c:
            raise SystemExit(f"grid file has duplicate driver codes: {dup_c}")
        gp = g["grid"][g["grid"] != 0]
        dup_g = gp[gp.duplicated()].unique().tolist()
        if dup_g:
            raise SystemExit(f"grid file has duplicate non-zero grid positions: {dup_g}")
        known = pd.concat([base, P.entrants_for(hist, SEASON, rnd - 1)]).drop_duplicates("code")
        ent = g.merge(known, on="code", how="left")
        missing = ent[ent["driver_id"].isna()]["code"].tolist()
        if missing:
            raise SystemExit(f"grid file codes not found in Jolpica entry lists: {missing}")
        absent = [c for c in base["code"] if c not in set(g["code"])]
        if absent:
            raise SystemExit(f"grid file is missing drivers {absent} (every entrant needs a row; pit-lane start = 0)")
        return ent[["driver_id", "code", "team_id", "team"]], ent["grid"].to_numpy(dtype=float), f"grid from {grid_file}"
    if len(q) == 0:
        raise SystemExit(
            f"grid mode: Jolpica has no qualifying for {SEASON} round {rnd} yet. "
            "Wait for it, pass --grid-file, or use --mode prequali."
        )
    # Jolpica's quali table omits drivers with no time, who still start the race.
    # Union with the previous round's race entrants; the extras start from the
    # back (after all qualifiers), in previous-round entry order.
    prev = P.entrants_for(hist, SEASON, rnd - 1)
    extra = prev[~prev["driver_id"].isin(base["driver_id"])]
    if len(extra):
        print(f"WARNING: no qualifying row for {extra['code'].tolist()}; added from round {rnd - 1} "
              "entrants and placed at the back of the grid.")
        base = pd.concat([base, extra], ignore_index=True)
    if len(base) != 22:
        print(f"WARNING: field has {len(base)} cars, expected 22.")
    ent = base.merge(q[["driver_id", "quali_position"]], on="driver_id", how="left")
    n = len(ent)
    gq = ent["quali_position"].to_numpy(dtype=float)
    nan = np.isnan(gq)
    gq[nan] = np.arange(n - nan.sum() + 1, n + 1)
    return ent[["driver_id", "code", "team_id", "team"]], gq, "grid = Jolpica qualifying order (NO penalties applied)"


def resolve(hist, rnd, mode_arg, grid_file, override_dir: Path | None = None):
    """Single decision point for CLI and API.

    Returns (mode, entry, grid, source_text, grid_source, has_quali). An existing override file
    counts as an explicit grid file, and (like --grid-file) switches auto mode to grid mode even
    if Jolpica has no qualifying. grid_source is None in prequali mode."""
    q = hist["qualifying"]
    has_q = len(q[(q["season"] == SEASON) & (q["round"] == rnd)]) > 0
    ov = override_grid_path(rnd, override_dir)
    if not grid_file and ov.exists():
        grid_file = str(ov)
    mode = mode_arg
    if mode == "auto":
        mode = "grid" if (has_q or grid_file) else "prequali"
    ent, grid, src = _entry_and_grid(hist, rnd, mode, grid_file)
    gs = None if mode == "prequali" else (GRID_OVERRIDE if grid_file else GRID_QUALI)
    return mode, ent, grid, src, gs, has_q


def print_table(t: pd.DataFrame, mode: str, title: str):
    print(f"\n{title}")
    gcol = "exp_grid" if mode == "prequali" else "grid"
    ghdr = "E[grid]" if mode == "prequali" else "Grid"
    if mode == "prequali" and "p_pole" in t.columns:
        print("qualifying: " + ", ".join(
            f"{r['code']} pole {100 * r['p_pole']:.0f}%" for _, r in t.sort_values("p_pole", ascending=False).head(4).iterrows()))
    print(f"{'#':>2}  {'Driver':<4} {'Team':<16} {ghdr:>7} {'P(win)':>7} {'P(pod)':>7} {'P(top10)':>8} {'P(DNF)':>7} {'E[pos]':>6}")
    for i, r in t.iterrows():
        print(
            f"{i + 1:>2}  {r['code']:<4} {str(r['team'])[:16]:<16} {r[gcol]:>7.1f} "
            f"{100 * r['p_win']:>6.1f}% {100 * r['p_podium']:>6.1f}% {100 * r['p_top10']:>7.1f}% "
            f"{100 * r['p_dnf']:>6.1f}% {r['exp_pos']:>6.1f}"
        )
    print(f"    sums: win {t['p_win'].sum():.3f}, podium {t['p_podium'].sum():.3f}, top10 {t['p_top10'].sum():.3f}")


def predict(args):
    rnd = args.round or next_round(args.offline)
    hist, _, sched = load(args.offline, rnd, with_fp=CONFIG.use_fp or args.fp)
    ev = sched[(sched["season"] == SEASON) & (sched["round"] == rnd)].iloc[0]
    mode, ent, grid, src, grid_source, has_q = resolve(hist, rnd, args.mode, args.grid_file)
    cfg = P.PredictorConfig(**{**CONFIG.__dict__, "n_sims": args.sims, "use_fp": CONFIG.use_fp or args.fp})
    t = P.predict_race(hist, SEASON, rnd, mode, cfg, entry_list=ent, grid=grid)
    st = t.attrs["stickiness"]
    if cfg.stick_gamma:
        print(f"track stickiness {st['circuit_id']}: {st['s_circuit']:.3f} vs global {st['s_global']:.3f} -> grid multiplier {st['mult']:.3f}")
    last = int(hist["results"].query("season == @SEASON")["round"].max())
    m = t.attrs["model"]
    title = (
        f"{ev['race_name']} {SEASON} (round {rnd}, {ev['circuit_id']}, race {ev['date']}) -- mode: {mode}\n"
        f"data: Jolpica {SEASON} rounds 1-{min(last, rnd - 1)} (+2022-25 for the rank table, 2025 coef prior); "
        f"{src}; grid_source: {grid_source}; qualifying for this round "
        f"{'AVAILABLE' if has_q else 'not yet published'}\n"
        f"model: b_race={m.b_race:.3f}, b_grid={m.b_grid:.3f}, a_quali={m.a_quali:.3f}, b_fp={m.b_fp:.3f}, "
        f"blend {1 - cfg.blend_baseline:.2f} PL model + {cfg.blend_baseline:.2f} "
        f"{'grid' if mode == 'grid' else 'championship'}-rank table, n_sims={cfg.n_sims}"
    )
    print_table(t, mode, title)
    hl = headline(t)
    print(f"headline: winner {hl['winner']}, podium {' '.join(hl['podium'])}")
    t["grid_source"] = grid_source  # saved so the API's stale fallback can still show the penalty note
    out = Path(args.out_dir) if getattr(args, "out_dir", None) else OUT_DIR / "predictions"
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    path = out / f"{SEASON}_R{rnd:02d}_{mode}_{stamp}.csv"
    t.to_csv(path, index=False)
    print(f"\nsaved {path}")


# --------------------------------------------------------------- evaluation

VARIANTS = {
    # PL model alone (blend 0): 2026-only vs each 2025 prior
    "PL 2026-only": dict(w2025_level2=0.0, driver_prior_2025=0.0, blend_baseline=0.0),
    "PL +2025 coef prior": dict(w2025_level2=1.0, driver_prior_2025=0.0, blend_baseline=0.0),
    "PL +2025 driver prior": dict(w2025_level2=0.0, driver_prior_2025=0.5, blend_baseline=0.0),
    "PL +2025 both": dict(w2025_level2=1.0, driver_prior_2025=0.5, blend_baseline=0.0),
    # blended with the rank-frequency table (the shipped config is "chosen")
    "blend 2026-only": dict(w2025_level2=0.0, driver_prior_2025=0.0),
    "chosen": dict(),
}
FP_VARIANT = {"chosen +FP": dict(use_fp=True)}


def evaluate(args):
    rnd = next_round(args.offline) if not args.round else args.round
    hist, spr, _ = load(args.offline, rnd, with_fp=args.fp)
    last = int(hist["results"].query("season == @SEASON")["round"].max())
    rounds = range(2, last + 1)
    rows, preds = [], {}
    base = {**CONFIG.__dict__, "n_sims": args.sims}
    for name in ("uniform", "grid", "champ", "form"):
        preds[f"baseline: {name}"] = P.walk_forward(hist, SEASON, rounds, name, CONFIG)
    variants = {**VARIANTS, **(FP_VARIANT if args.fp else {})}
    for vname, v in variants.items():
        cfg = P.PredictorConfig(**{**base, **v})
        for mode in ("prequali", "grid"):
            preds[f"model {mode} ({vname})"] = P.walk_forward(hist, SEASON, rounds, f"model_{mode}", cfg)
    for k, wf in preds.items():
        for label, rr in (("R2-%d" % last, rounds), ("R6-%d" % last, range(6, last + 1))):
            s = P.score(wf[wf["round"].isin(list(rr))])
            rows.append({"rounds": label, "method": k, **s})
    tab = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    for label in tab["rounds"].unique():
        print(f"\nWalk-forward {SEASON}, {label} (lower is better; *_logloss/brier = mean per race of per-driver binary scores; winner_nll = -log P(actual winner))")
        print(tab[tab["rounds"] == label].drop(columns=["rounds"]).set_index("method").round(4).to_string())
    chosen_pre = f"model prequali ({args.variant})"
    chosen_grid = f"model grid ({args.variant})"
    for k in (chosen_pre, chosen_grid, "baseline: grid", "baseline: champ"):
        for ev in ("win", "podium"):
            print(f"\nCalibration, {k}, {ev} (pooled R2-{last}):")
            print(P.calibration_table(preds[k], ev).round(3).to_string())
    g = hist["results"].query("season == @SEASON")[["round", "driver_id", "grid"]]
    for k in (chosen_grid, "baseline: grid"):
        wf = preds[k].merge(g, on=["round", "driver_id"])
        pole = wf[wf["grid"] == 1]
        print(f"\n{k}: mean P(win | P1 start) = {pole['p_win'].mean():.3f}, observed {pole['y_win'].mean():.3f} (n={len(pole)})")
    out = OUT_DIR / "eval"
    out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out / f"walk_forward_{SEASON}.csv", index=False)
    pd.concat([d.assign(method=k) for k, d in preds.items()]).to_csv(out / f"walk_forward_{SEASON}_predictions.csv", index=False)
    print(f"\nsaved {out / f'walk_forward_{SEASON}.csv'}")


def evaluate_quali(args):
    """T-035: walk-forward qualifying predictions (pole / Q3 log-loss, Spearman of exp_grid)
    vs previous-race quali order, championship order and uniform. 2026 and 2025."""
    rnd = next_round(args.offline) if not args.round else args.round
    hist, _, _ = load(args.offline, rnd)
    cfg = P.PredictorConfig(**{**CONFIG.__dict__, "n_sims": args.sims})
    rows = []
    for season in (2026, 2025):
        last = int(hist["qualifying"].query("season == @season")["round"].max())
        for method in ("model",) + P.QUALI_BASELINES:
            s = P.quali_score(P.quali_walk_forward(hist, season, range(2, last + 1), method, cfg))
            rows.append({"season": season, "rounds": f"R2-{last}", "method": method, **s})
    tab = pd.DataFrame(rows)
    print(tab.round(4).to_string(index=False))
    out = OUT_DIR / "eval"
    out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out / "quali_walk_forward.csv", index=False)
    print(f"\nsaved {out / 'quali_walk_forward.csv'}")


def by_race(args):
    rnd = next_round(args.offline) if not args.round else args.round
    hist, _, _ = load(args.offline, rnd)
    last = int(hist["results"].query("season == @SEASON")["round"].max())
    cfg = P.PredictorConfig(**{**CONFIG.__dict__, "n_sims": args.sims})
    t = P.by_race_table(hist, SEASON, range(1, last + 1), cfg, "grid")
    out = OUT_DIR / "predictions"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"walk_forward_{SEASON}_by_race.csv"
    t.to_csv(path, index=False)
    pd.set_option("display.width", 250)
    print(t.drop(columns=["circuit_id", "stick_n_current_season", "stick_n_earlier_seasons"]).round(3).to_string(index=False))
    print(f"\nsaved {path}")


def main():
    global CONFIG
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["auto", "prequali", "grid"], default="auto")
    ap.add_argument("--round", type=int, default=None, help="target round (default: next unraced 2026 round)")
    ap.add_argument("--grid-file", default=None, help="CSV with columns code,grid (post-penalty; pit lane = 0)")
    ap.add_argument("--out-dir", default=None, help="directory for the saved prediction CSV (default data/jolpica/predictions)")
    ap.add_argument("--evaluate", action="store_true", help="walk-forward evaluation vs baselines")
    ap.add_argument("--evaluate-quali", action="store_true", help="T-035: walk-forward qualifying evaluation -> eval/quali_walk_forward.csv")
    ap.add_argument("--variant", default="chosen", choices=list(VARIANTS) + list(FP_VARIANT),
                    help="model variant to show calibration for")
    ap.add_argument("--fp", action="store_true",
                    help="also pull OpenF1 practice long runs (feature; --evaluate adds a 'chosen +FP' variant)")
    ap.add_argument("--by-race", action="store_true", help="T-014: per-race walk-forward table (grid mode) -> predictions/walk_forward_2026_by_race.csv")
    ap.add_argument("--stick-gamma", type=float, default=None, help="override PredictorConfig.stick_gamma (0 = off)")
    ap.add_argument("--dnf-track-m", type=float, default=None, help="T-029: track-level DNF shrinkage m (0 = off, the default)")
    ap.add_argument("--offline", action="store_true", help="use cached Jolpica data only")
    ap.add_argument("--sims", type=int, default=CONFIG.n_sims)
    args = ap.parse_args()
    if args.stick_gamma is not None:
        CONFIG = P.PredictorConfig(**{**CONFIG.__dict__, "stick_gamma": args.stick_gamma})
    if args.dnf_track_m is not None:
        CONFIG = P.PredictorConfig(**{**CONFIG.__dict__, "dnf_track_m": args.dnf_track_m})
    if args.by_race:
        by_race(args)
    elif args.evaluate_quali:
        evaluate_quali(args)
    elif args.evaluate:
        evaluate(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
