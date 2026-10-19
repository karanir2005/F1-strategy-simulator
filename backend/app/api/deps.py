"""Shared API helpers (kept out of the route modules to avoid circular imports)."""

import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi import HTTPException

from app.core.track_params import TrackParamsUnavailable, get_track, load_tracks


def tracks_or_503() -> List[dict]:
    try:
        return load_tracks()
    except TrackParamsUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))


def resolve_track(track_id: str) -> dict:
    """Track dict, or HTTPException: 503 if the params file is missing, 404 for an unknown id."""
    tracks_or_503()
    track = get_track(track_id)
    if track is None:
        raise HTTPException(status_code=404, detail=f"Unknown track_id '{track_id}'")
    return track
