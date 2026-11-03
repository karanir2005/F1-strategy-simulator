"""Scorecard (LORO) A/B (T-047: pace_ordered vs pace_ordered + durability=stint_lengths). Not written to data/scorecard."""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "backend"))
import validate_scorecard as vs
out = {}
for mode in ("pace_ordered", "durable"):
    card = vs.run_scorecard(model_kwargs={"offset_mode": "pace_ordered", **({"durability": "stint_lengths"} if mode == "durable" else {"durability": None})})
    out[mode] = {k: card[k] for k in card if k != "per_race"}
    print("=====", mode); vs.print_scorecard(card)
Path(ROOT / "data" / "strategy_eval" / "scorecard_ab.json").write_text(json.dumps(out, indent=1, default=vs._json_default), encoding="utf-8")
