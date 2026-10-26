"""T-045: ranking-statistic comparison + inversion audit on eval/rank_audit_drivers.csv."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

D = pd.read_csv(Path(__file__).parent / "rank_audit_drivers.csv")

STATS = {  # name -> (column(s) sort key, ascending flags)
    "exp_pos": (["exp_pos"], [True]),
    "med_pos": (["med_pos", "exp_pos"], [True, True]),
    "exp_pos_cls": (["exp_pos_cls"], [True]),
    "exp_pts": (["exp_pts"], [False]),
    "pwin_then_exppos": (["p_win", "exp_pos"], [False, True]),
    "p_podium": (["p_podium", "exp_pos"], [False, True]),
}


def order(g, stat):
    cols, asc = STATS[stat]
    return g.sort_values(cols, ascending=asc, kind="stable")


def inversions(g, stat):
    """Count pairs (A ranked below B) where A is clearly better by the model's own probabilities:
    p_win_A >= p_win_B + .15 and p_podium_A >= p_podium_B (a 'win' inversion), plus favourite>50% not first."""
    o = order(g, stat).reset_index(drop=True)
    n = 0
    bad = []
    for i in range(len(o)):
        for j in range(i + 1, len(o)):  # i ranked above j
            b, a = o.loc[i], o.loc[j]   # a is lower-ranked
            if a["p_win"] >= b["p_win"] + 0.15 and a["p_podium"] >= b["p_podium"]:
                n += 1
                bad.append((a["code"], b["code"]))
    fav = g.loc[g["p_win"].idxmax()]
    fav_bad = fav["p_win"] > 0.5 and o.iloc[0]["code"] != fav["code"]
    return n, bad, bool(fav_bad)


def metrics(df, stat):
    sp, mae, t3, t1, inv, favb, podhit = [], [], [], [], 0, 0, []
    for (s, r), g in df.groupby(["season", "round"]):
        o = order(g, stat).reset_index(drop=True)
        o["rk"] = np.arange(1, len(o) + 1)
        sp.append(o["rk"].corr(o["actual_position"], method="spearman"))
        mae.append((o["rk"] - o["actual_position"]).abs().mean())
        pod = set(o["code"][:3]) & set(g.loc[g["actual_position"] <= 3, "code"])
        podhit.append(len(pod))
        t1.append(float(o["actual_position"].iloc[0] == 1))
        n, _, fb = inversions(g, stat)
        inv += n
        favb += fb
    return dict(spearman=np.mean(sp), mae=np.mean(mae), podium_hits=np.mean(podhit), winner_hit=np.mean(t1),
                inversions=inv, fav_not_first=favb, races=len(sp))


if __name__ == "__main__":
    rows = []
    for season, g in D.groupby("season"):
        for stat in STATS:
            rows.append(dict(season=season, stat=stat, **metrics(g, stat)))
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(t.round(3).to_string(index=False))
