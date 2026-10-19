"""
Build staging CSVs for every held 2025 + 2026 race from OpenF1 + Jolpica (T-009).

    python pull_openf1.py                 # all held races -> data/staging/
    python pull_openf1.py --only 2026:Japan 2025:Monaco
    python pull_openf1.py --out data/staging --offline   # cache only, no network

Raw API responses are cached under data/openf1/raw/ (never refetched).
FastF1 is not used. See backend/app/core/openf1_loader.py.
"""
import argparse
import sys
import warnings

sys.path.insert(0, "backend")
warnings.filterwarnings("ignore", category=FutureWarning)

from app.core import openf1_loader as o  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", default=o.STAGING_DIR)
    ap.add_argument("--only", nargs="*", help="season:label pairs, e.g. 2026:Japan")
    ap.add_argument("--today", default=None, help="YYYY-MM-DD cutoff (default: today)")
    args = ap.parse_args()

    races = o.list_races(today=args.today)
    if args.only:
        want = {tuple(x.split(":")) for x in args.only}
        races = [r for r in races if (str(r["season"]), r["label"]) in want]
    for m in races:
        print(f"{m['season']} R{m['round']:02d} {m['label']:<12} key={m['session_key']}", flush=True)
        t = o.build_race(m)
        o.write_race(t, args.out)
        L = t["laps"]
        print(f"    {len(L)} laps, {L['driver'].nunique()} drivers, "
              f"{int(L['is_tyre_change'].sum())} tyre changes", flush=True)


if __name__ == "__main__":
    main()
