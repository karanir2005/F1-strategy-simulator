"""
Actual team tyre strategies vs. what the optimizer recommends (T-043).

Run from the repo root:
    .venv/Scripts/python.exe data/strategy_eval/eval_strategy.py [--variant baseline] [--n-sims 300]

For every DRY, red-flag-free race with lap data (2026: data/processed + the OpenF1-sourced
Austin R18; 2025: data/staging usable races) it

  1. extracts the real stints of the top-10 finishers (tyre_stint splits: a stop onto the same
     compound counts as a stop) and the most common top-10 strategy;
  2. fits the tyre model LEAVE-ONE-RACE-OUT, era-matched (2026 races use the other 2026 races,
     2025 races the other 2025 races; Austin R18 uses all 16 earlier 2026 races), so the
     judged race never fits its own model;
  3. runs optimizer.optimize() with that model and CLEAN stint caps (pooled p90 stint length
     per compound over the other races of the same era -- the shipped track caps come from the
     very race being judged, which would leak the answer);
  4. scores optimizer-#1 vs the real top-10 (stop count, compound sequence, rank/gap of the
     winner's real plan, pit-window error).

Variants (--variant) switch on the Phase-2 changes in this file; see VARIANTS.
Outputs: data/strategy_eval/strategy_vs_actual[_<variant>].csv and a summary JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.core import tire_model as tm  # noqa: E402
from app.core.optimizer import optimize  # noqa: E402
from app.core.race_simulator import Stint, simulate_race  # noqa: E402

OUT_DIR = ROOT / "data" / "strategy_eval"
TRACKS = {t["track_id"]: t for t in json.loads((ROOT / "data" / "tracks" / "track_params.json").read_text(encoding="utf-8"))}
RAW = ROOT / "data" / "openf1_predictor" / "raw"
L = {"SOFT": "S", "MEDIUM": "M", "HARD": "H"}
AUSTIN26 = "Austin_2026"


# ------------------------------------------------------------------ data ----
def key_of(race_file: str) -> str:
    return Path(race_file).name.replace("_laps.csv", "")


def driver_stints(laps: pd.DataFrame, driver: str) -> list:
    d = laps[laps["driver"] == driver].sort_values("lap_number")
    out = []
    for _, g in d.groupby("tyre_stint", sort=True):
        comp = g["compound"].dropna()
        c = comp.iloc[0] if len(comp) else "UNKNOWN"
        out.append((c, int(g["lap_number"].max() - g["lap_number"].min() + 1)))
    return out


def austin26_strategies() -> tuple:
    """OpenF1 /stints + /drivers for session 11396 + Jolpica R18 results -> ({code: [(compound, laps)]}, order)."""
    st = json.loads((RAW / "stints_11396.json").read_text(encoding="utf-8"))
    code = {d["driver_number"]: d["name_acronym"] for d in json.loads((RAW / "drivers_11396.json").read_text(encoding="utf-8"))}
    res = pd.read_csv(ROOT / "data" / "jolpica" / "derived" / "results.csv")
    res = res[(res.season == 2026) & (res["round"] == 18) & (res.session == "race")].sort_values("position")
    strat = {}
    for s in sorted(st, key=lambda s: (s["driver_number"], s["stint_number"])):
        strat.setdefault(code[s["driver_number"]], []).append((s["compound"], s["lap_end"] - s["lap_start"] + 1))
    return strat, res["code"].tolist()


def load_races() -> tuple:
    """-> (races dict key -> info, skipped list). info: era, strategies {driver: stints}, order, total_laps, sc_frac."""
    races, skipped = {}, []
    files = [(p, "2026_regs") for p in sorted((ROOT / "data" / "processed").glob("*_2026_laps.csv"))]
    files += [(p, "2025_regs") for p in sorted((ROOT / "data" / "staging").glob("*_2025_laps.csv"))]
    for p, era in files:
        key = key_of(p.name)
        if era == "2025_regs" and key in tm.EXCLUDED_2025_RACES:
            skipped.append((key, "wet / missing laps (tire_model.EXCLUDED_2025_RACES)"))
            continue
        info_p = p.with_name(f"{key}_race_info.json")
        if info_p.exists() and not (json.loads(info_p.read_text(encoding="utf-8")).get("data_quality") or {}).get("usable", True):
            skipped.append((key, "OpenF1 data unusable"))
            continue
        laps = pd.read_csv(p)
        if laps["has_red"].astype(bool).any():
            skipped.append((key, "red flag (free tyre changes, stints not comparable)"))
            continue
        if laps["compound"].isin(["INTERMEDIATE", "WET"]).mean() > 0.05 or laps.get("rainfall", pd.Series(False)).astype(bool).mean() > 0.1:
            skipped.append((key, "wet / mixed conditions"))
            continue
        res = pd.read_csv(p.with_name(f"{key}_results.csv")).sort_values("position")
        strat = {d: driver_stints(laps, d) for d in laps["driver"].unique()}
        total = int(laps["lap_number"].max())
        sc = float((laps["has_sc"].astype(bool) | laps["has_vsc"].astype(bool)).groupby(laps["lap_number"]).any().mean())
        neut = laps.groupby("lap_number")["is_neutralised"].apply(lambda x: x.astype(bool).mean())
        races[key] = dict(era=era, laps=laps, strat=strat, order=res["driver"].tolist(), total_laps=total, sc_frac=sc,
                          race_file=str(p), neutral=set(int(l) for l in neut.index[neut > 0.5]))
    # Austin R18 (OpenF1 stints, no laps CSV)
    strat, order = austin26_strategies()
    # no flag data for R18: neutralised lap = field-median lap >= 8% slower than the race median (lap 1 excluded)
    ol = pd.DataFrame(json.loads((RAW / "laps_11396.json").read_text(encoding="utf-8")))
    med = ol.groupby("lap_number")["lap_duration"].median()
    neutral = set(int(l) for l in med.index[(med > 1.08 * med.median()) & (med.index > 1)])
    races[AUSTIN26] = dict(era="2026_regs", laps=None, strat=strat, order=order, total_laps=56,
                           sc_frac=len(neutral) / 56, race_file=None, neutral=neutral)
    return races, skipped


def stint_caps(races: dict, era: str, exclude: str, pct: float = 90, top10_only: bool = False) -> tuple:
    """Pooled p<pct> stint length per compound over the other races of the era (clean, no leakage), plus max."""
    lens = {c: [] for c in L}
    for k, r in races.items():
        if k == exclude or r["era"] != era:
            continue
        for d, s in r["strat"].items():
            if top10_only and d not in r["order"][:10]:
                continue
            for c, n in s:
                if c in lens and n >= 5:
                    lens[c].append(n)
    p90 = {c: int(np.percentile(v, pct)) for c, v in lens.items() if len(v) >= 5}
    mx = {c: int(max(v)) for c, v in lens.items() if len(v) >= 5}
    return p90, mx


def fallback_track(r: dict) -> dict:
    """Circuits absent from track_params.json (Bahrain, Saudi, Imola 2025): pit loss and a base lap from the race itself.
    Neither affects plan RANKING (base lap is a constant; pit loss is a track property, median of clean stops)."""
    laps = r["laps"]
    ok = laps[laps["is_tyre_change"].astype(bool) & ~laps["pit_under_sc"].astype(bool) & ~laps["pit_under_vsc"].astype(bool)]
    pl = ok["pit_lane_s"].dropna()
    pl = pl[(pl >= 10) & (pl <= 60)]
    green = laps[laps["is_green_flag"].astype(bool)]["lap_time_seconds"].dropna()
    return dict(pit_loss_s=float(pl.median()) if len(pl) >= 5 else 23.5, base_laptime_s=float(np.percentile(green, 10)),
                max_stint_laps={}, typical_stint_laps={})


# ----------------------------------------------------------------- scoring --
def seq_of(s) -> tuple:
    return tuple(L.get(c, "?") for c, _ in s)


def pit_laps(s) -> list:
    return [int(x) for x in np.cumsum([n for _, n in s])[:-1]]


def fmt(s) -> str:
    return "/".join(f"{L.get(c, '?')}{n}" for c, n in s)


def det_time(plan, T, model) -> float:
    """Model time of a plan [(compound, laps)] with the last stint stretched/shrunk to the full distance."""
    plan = [[c, n] for c, n in plan]
    plan[-1][1] += T["total_laps"] - sum(n for _, n in plan)
    if plan[-1][1] < 1:
        return float("nan")
    return simulate_race([Stint(c, n) for c, n in plan], T["base_laptime_s"], model, T["pit_loss_s"]).total_time


def score_race(key, r, model, T, res, n_top=10) -> dict:
    top = [d for d in r["order"][:n_top] if d in r["strat"]]
    strats = {d: r["strat"][d] for d in top}
    seqs = Counter(seq_of(s) for s in strats.values())
    stops = Counter(len(s) - 1 for s in strats.values())
    # modal strategy = most common compound sequence; ties -> the winner's if tied, else fewer stops
    best_n = max(seqs.values())
    cands = [q for q, n in seqs.items() if n == best_n]
    win_seq = seq_of(strats[top[0]])
    modal = win_seq if win_seq in cands else sorted(cands, key=lambda q: (len(q), q))[0]
    modal_stops = max(stops.items(), key=lambda kv: (kv[1], -kv[0]))[0]
    def n_green_stops(strat):
        ends = np.cumsum([n for _, n in strat])[:-1]
        return int(sum(1 for e in ends if e not in r["neutral"] and e + 1 not in r["neutral"]))

    plans = res.plans
    p1 = plans[0]
    p1_plan = [(s["compound"], s["laps"]) for s in p1["stints"]]
    p1_seq, p1_stops = seq_of(p1_plan), p1["n_stops"]
    win = strats[top[0]]
    win_key = (len(win) - 1, tuple(sorted(seq_of(win))))
    rank = next((p["rank"] for p in plans if (p["n_stops"], tuple(sorted(L[s["compound"]] for s in p["stints"]))) == win_key), None)
    gap = det_time(win, T, model) - det_time(p1_plan, T, model)
    # pit-window error vs top-10 drivers that made the same number of stops as the optimizer's #1
    same = [pit_laps(s) for s in strats.values() if len(s) - 1 == p1_stops]
    pit_err = None
    if same:
        med = np.median(np.array(same, dtype=float), axis=0)
        pit_err = float(np.mean(np.abs(np.array(p1["pit_laps"], dtype=float) - med)))
    gsame = [pit_laps(s) for s in strats.values() if len(s) - 1 == p1_stops and n_green_stops(s) == p1_stops]
    pit_err_green = None
    if gsame:
        gmed = np.median(np.array(gsame, dtype=float), axis=0)
        pit_err_green = float(np.mean(np.abs(np.array(p1["pit_laps"], dtype=float) - gmed)))
    pit_err_win = float(np.mean(np.abs(np.array(p1["pit_laps"]) - np.array(pit_laps(win))))) if len(win) - 1 == p1_stops else None
    act_laps = Counter()
    for s in strats.values():
        for c, n in s:
            act_laps[c] += n
    tot = sum(act_laps.values())
    opt_soft = sum(n for c, n in p1_plan if c == "SOFT") / T["total_laps"]
    top3 = [(p["n_stops"], tuple(sorted(L[s["compound"]] for s in p["stints"]))) for p in plans[:3]]
    modal_key = (len(modal) - 1, tuple(sorted(modal)))
    green = Counter(n_green_stops(s) for s in strats.values())
    green_modal = max(green.items(), key=lambda kv: (kv[1], -kv[0]))[0]
    n_sc_stops = sum((len(s) - 1) - n_green_stops(s) for s in strats.values())
    return dict(
        race=key, era=r["era"], total_laps=T["total_laps"], sc_frac=round(r["sc_frac"], 3) if r["sc_frac"] == r["sc_frac"] else None,
        winner=top[0], winner_strategy=fmt(win),
        top10_strategies=" | ".join(f"{d}:{fmt(s)}" for d, s in strats.items()),
        top10_stops=" ".join(f"{k}x{v}" for k, v in sorted(stops.items())),
        modal_seq="-".join(modal), modal_stops=modal_stops, modal_seq_count=seqs[modal],
        modal_pit_laps=str(pit_laps(next(s for s in strats.values() if seq_of(s) == modal))),
        opt_top_plan=fmt(p1_plan), opt_seq="-".join(p1_seq), opt_stops=p1_stops, opt_pit_laps=str(p1["pit_laps"]),
        opt_top3=" | ".join(fmt([(s["compound"], s["laps"]) for s in p["stints"]]) for p in plans[:3]),
        stops_match_modal=int(p1_stops == modal_stops),
        modal_green_stops=green_modal, stops_match_modal_green=int(p1_stops == green_modal), top10_stops_under_sc=n_sc_stops,
        neutral_laps=len(r["neutral"]), seq_match_modal=int(p1_seq == modal),
        set_match_modal=int(tuple(sorted(p1_seq)) == tuple(sorted(modal))),
        modal_in_opt_top3=int(modal_key in top3),
        stops_match_winner=int(p1_stops == len(win) - 1), seq_match_winner=int(p1_seq == win_seq),
        winner_rank_in_opt=rank, winner_gap_s=round(gap, 1),
        pit_err_vs_top10_same_stops=None if pit_err is None else round(pit_err, 1),
        pit_err_green_stops_only=None if pit_err_green is None else round(pit_err_green, 1), n_drivers_green_same_stops=len(gsame),
        pit_err_vs_winner=None if pit_err_win is None else round(pit_err_win, 1),
        act_soft_share=round(act_laps["SOFT"] / tot, 3), act_med_share=round(act_laps["MEDIUM"] / tot, 3),
        act_hard_share=round(act_laps["HARD"] / tot, 3),
        opt_soft_share=round(opt_soft, 3),
        opt_med_share=round(sum(n for c, n in p1_plan if c == "MEDIUM") / T["total_laps"], 3),
        opt_hard_share=round(sum(n for c, n in p1_plan if c == "HARD") / T["total_laps"], 3),
        caps=json.dumps(res.effective_caps),
    )


# ---------------------------------------------------------------- variants ---
# Phase-2 switches. Each variant is a dict of knobs read by run().
VARIANTS = {
    "baseline": {"model_kwargs": {"offset_mode": "joint"}},  # T-020 shipped tyre fit (before T-043)
    # Change A (T-043): pace offsets from the free fit projected onto the pace ordering only
    "pace_ordered": {"model_kwargs": {"offset_mode": "pace_ordered"}},
    # Change B candidate: + circuit wear borrowed from the other era (tire_model.transfer_track_wear)
    "pace_ordered_sc": {"model_kwargs": {"offset_mode": "pace_ordered"}, "sc_on": True},
    "pace_ordered_xwear": {"model_kwargs": {"offset_mode": "pace_ordered"}, "transfer_wear": True},
}


_OTHER = {}


def OTHER_MODELS(era: str, model_kwargs: dict):
    """Full-data tyre model of the other regulation era (cached): its circuits' relative wear is the prior for tracks
    the target era has not raced. It never contains the judged race (different season)."""
    k = (era, json.dumps(model_kwargs, sort_keys=True))
    if k not in _OTHER:
        _OTHER[k] = tm.build_tire_model(laps=tm.load_all_laps(eras=(era,)), eras=(era,), **model_kwargs)
    return _OTHER[k]


def apply_overrides(model: dict, cfg: dict, races: dict, era: str, key: str) -> dict:
    """Phase-2 experiment knobs applied to a copy of the LORO tyre model."""
    m = dict(model)
    if "offsets" in cfg:
        m["_compound_offsets"] = dict(m["_compound_offsets"], **cfg["offsets"])
    if "life_pct" in cfg:  # stint-life cliff: beyond the p<life_pct> observed stint length add cliff_k s/lap per lap
        life, _ = stint_caps(races, era, exclude=key, pct=cfg["life_pct"])
        m["_cliff_penalties"] = {c: cfg["cliff_k"] for c in L2}
        for c in L2:
            if c in m and c in life:
                m[c] = dict(m[c], max_reliable_age=min(m[c]["max_reliable_age"], life[c]), cliff_age=None)
    return m


L2 = ("SOFT", "MEDIUM", "HARD")


def run(variant: str, n_sims: int, caps_mode: str = "clean", only: list = None) -> pd.DataFrame:
    cfg = variant if isinstance(variant, dict) else VARIANTS[variant]
    races, skipped = load_races()
    all_laps = {era: tm.load_all_laps(eras=(era,)) for era in ("2026_regs", "2025_regs")}
    all_laps["2026_regs"] = all_laps["2026_regs"][~all_laps["2026_regs"]["race_file"].str.contains("_2025_")]
    rows, rows_lam = [], {}
    for key, r in races.items():
        if only and key not in only:
            continue
        era = r["era"]
        train = all_laps[era]
        if r["race_file"] is not None:
            train = train[train["race_file"].map(lambda f: key_of(f)) != key]
        model = tm.build_tire_model(laps=train, eras=(era,), **cfg.get("model_kwargs", {}))
        tid = tm.track_id_of_race(key)
        T = dict(TRACKS[tid]) if tid in TRACKS else fallback_track(r)
        T["total_laps"] = r["total_laps"]
        if caps_mode == "clean":
            p90, mx = stint_caps(races, era, exclude=key, pct=cfg.get("cap_pct", 90), top10_only=cfg.get("cap_top10", False))
            maxc, typ = mx, p90
            model = apply_overrides(model, cfg, races, era, key)
        else:  # shipped track-specific caps (leak for tracks whose source race is the judged race)
            maxc, typ = T["max_stint_laps"], T.get("typical_stint_laps")
            if len(maxc) < 2:  # circuit absent from track_params.json (Bahrain/Imola/Saudi 2025): no shipped caps
                continue
        wear_tid = None
        opt_kwargs = dict(cfg.get("opt_kwargs", {}))
        if cfg.get("sc_on"):
            from app.core.safety_cars import load_params
            opt_kwargs["safety_cars"] = load_params()
        if cfg.get("transfer_wear"):
            other_era = "2025_regs" if era == "2026_regs" else "2026_regs"
            om = OTHER_MODELS(other_era, cfg.get("model_kwargs", {}))
            model, lam = tm.transfer_track_wear(model, om, tid)
            rows_lam[key] = lam
            wear_tid = tid if lam is not None else None
        res = optimize(model, T["total_laps"], T["pit_loss_s"], T["base_laptime_s"], maxc, typical_stint_laps=typ,
                       max_stops=3, n_sims=n_sims, top_n=20, track_id=wear_tid, **opt_kwargs)
        rows.append(score_race(key, r, model, T, res))
        if not cfg.get("quiet"):
            print(f"{key:<16} opt {rows[-1]['opt_top_plan']:<22} modal {rows[-1]['modal_seq']:<8} winner {rows[-1]['winner_strategy']}")
    df = pd.DataFrame(rows)
    if rows_lam:
        df["wear_lambda"] = df["race"].map(rows_lam)
    df.attrs["skipped"] = skipped
    return df


def summarise(df: pd.DataFrame) -> dict:
    def block(d):
        return dict(
            n_races=int(len(d)),
            stops_match_modal=f"{int(d.stops_match_modal.sum())}/{len(d)}",
            stops_match_modal_green=f"{int(d.stops_match_modal_green.sum())}/{len(d)}",
            one_stop_always_baseline=f"{int((d.modal_stops == 1).sum())}/{len(d)}",
            seq_match_modal=f"{int(d.seq_match_modal.sum())}/{len(d)}",
            set_match_modal=f"{int(d.set_match_modal.sum())}/{len(d)}",
            modal_in_opt_top3=f"{int(d.modal_in_opt_top3.sum())}/{len(d)}",
            stops_match_winner=f"{int(d.stops_match_winner.sum())}/{len(d)}",
            median_winner_gap_s=round(float(d.winner_gap_s.median()), 1),
            median_winner_rank=None if d.winner_rank_in_opt.dropna().empty else float(d.winner_rank_in_opt.dropna().median()),
            winner_in_opt_top20=f"{int(d.winner_rank_in_opt.notna().sum())}/{len(d)}",
            median_pit_err_green_laps=None if d.pit_err_green_stops_only.dropna().empty else float(d.pit_err_green_stops_only.median()),
            n_pit_err_green=int(d.pit_err_green_stops_only.notna().sum()),
            median_pit_err_laps=None if d.pit_err_vs_top10_same_stops.dropna().empty else float(d.pit_err_vs_top10_same_stops.median()),
            mean_opt_stops=round(float(d.opt_stops.mean()), 2), mean_modal_stops=round(float(d.modal_stops.mean()), 2),
            soft_share_opt=round(float(d.opt_soft_share.mean()), 3), soft_share_actual=round(float(d.act_soft_share.mean()), 3),
            med_share_opt=round(float(d.opt_med_share.mean()), 3), med_share_actual=round(float(d.act_med_share.mean()), 3),
            hard_share_opt=round(float(d.opt_hard_share.mean()), 3), hard_share_actual=round(float(d.act_hard_share.mean()), 3),
        )
    out = {"all": block(df)}
    for era, d in df.groupby("era"):
        out[era] = block(d)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="pace_ordered", choices=list(VARIANTS))
    ap.add_argument("--n-sims", type=int, default=300)
    ap.add_argument("--caps", default="clean", choices=["clean", "shipped"])
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    df = run(a.variant, a.n_sims, a.caps, a.only)
    tag = a.variant + ("" if a.caps == "clean" else "_shippedcaps")
    name = "strategy_vs_actual.csv" if tag == "pace_ordered" else f"strategy_vs_actual_{tag}.csv"
    df.to_csv(OUT_DIR / name, index=False)
    summ = summarise(df)
    summ["skipped"] = df.attrs["skipped"]
    (OUT_DIR / name.replace(".csv", "_summary.json")).write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
