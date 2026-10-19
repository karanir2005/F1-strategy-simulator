"""
Loader for per-track parameters (data/tracks/track_params.json, built offline
by build_track_params.py -- see team/contracts/track-optimizer.md section 1).

One loader function so tests and deployments can point at a different file:
set the TRACK_PARAMS_PATH env var, or call set_track_params_path(). The file
is read once per path and cached; a missing/invalid file raises
TrackParamsUnavailable, which the API turns into a 503 with a clear message.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_TRACK_PARAMS_PATH = Path(__file__).resolve().parents[3] / "data" / "tracks" / "track_params.json"

_path_override: Optional[Path] = None
_cache: Dict[str, List[dict]] = {}


class TrackParamsUnavailable(RuntimeError):
    """The track parameters file is missing or malformed."""


def set_track_params_path(path: Optional[Path]) -> None:
    """Override the file location (None restores the default). Clears the cache."""
    global _path_override
    _path_override = Path(path) if path is not None else None
    _cache.clear()


def get_track_params_path() -> Path:
    if _path_override is not None:
        return _path_override
    env = os.environ.get("TRACK_PARAMS_PATH")
    return Path(env) if env else DEFAULT_TRACK_PARAMS_PATH


def load_tracks() -> List[dict]:
    path = get_track_params_path()
    key = str(path)
    if key not in _cache:
        if not path.exists():
            raise TrackParamsUnavailable(
                f"Track parameters file not found at {path}. "
                "Run build_track_params.py to generate data/tracks/track_params.json."
            )
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise TrackParamsUnavailable(f"Track parameters file {path} is not valid JSON: {exc}")
        if not isinstance(data, list):
            raise TrackParamsUnavailable(f"Track parameters file {path} must contain a JSON list")
        _cache[key] = data
    return _cache[key]


def get_track(track_id: str) -> Optional[dict]:
    """The track dict for track_id, or None if unknown."""
    for track in load_tracks():
        if track.get("track_id") == track_id:
            return track
    return None
