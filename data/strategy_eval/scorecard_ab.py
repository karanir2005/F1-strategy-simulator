"""Scorecard (LORO, 16 races) with offset_mode joint (shipped T-020) vs pace_ordered (T-043). Not written to data/scorecard."""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "backend"))
import validate_scorecard as vs
out = {}
for mode in ("joint", "pace_ordered"):
    card = vs.run_scorecard(model_kwargs={"offset_mode": mode})
    out[mode] = {k: card[k] for k in card if k != "per_race"}
    print("=====", mode); vs.print_scorecard(card)
Path(ROOT / "data" / "strategy_eval" / "scorecard_ab.json").write_text(json.dumps(out, indent=1, default=vs._json_default), encoding="utf-8")
