"""
Write per-track tyre wear (T-028) into data/tracks/track_params.json as the
additive optional field `tire_wear`:

    "tire_wear": {"SOFT": {"age_coef": 0.049, "global_age_coef": 0.037, "n_stints": 4, "pooled": false}, ...}

age_coef is the partially pooled linear wear slope in s per lap of tyre age
(tire_model.fit_track_wear, prior sd TRACK_WEAR_TAU, ordering SOFT>=MEDIUM>=HARD
kept); pooled=true means the track had < MIN_TRACK_STINTS stints of that
compound so the value IS the global slope. Only circuits with a 2026 race in
data/processed get the field; every other key in the file is untouched.

Run from the repo root:  python fit_track_wear_params.py
NOTE: build_track_params.py rewrites the whole file and does not know about
this field -- re-run this script after it.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))

from app.core import tire_model as tm  # noqa: E402

PATH = ROOT / "data" / "tracks" / "track_params.json"


def main() -> None:
    model = tm.build_tire_model()
    wear = model["_track_wear"]
    tracks = json.loads(PATH.read_text(encoding="utf-8"))
    for t in tracks:
        w = wear.get(t["track_id"])
        if not w:
            t.pop("tire_wear", None)
            continue
        t["tire_wear"] = {
            c: {
                "age_coef": round(v["age_coef"], 5),
                "global_age_coef": round(v["global_age_coef"], 5),
                "n_stints": v["n_stints"],
                "pooled": v["pooled"],
            }
            for c, v in w.items()
        }
    PATH.write_text(json.dumps(tracks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"tire_wear written for {sum('tire_wear' in t for t in tracks)}/{len(tracks)} tracks (tau={tm.TRACK_WEAR_TAU})")


if __name__ == "__main__":
    main()
