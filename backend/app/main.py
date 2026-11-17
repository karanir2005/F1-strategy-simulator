"""
FastAPI app entrypoint.

Run from the backend/ directory:
    uvicorn app.main:app --reload

Or for a quick standalone smoke test from anywhere:
    python backend/app/main.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.chat import router as chat_router
from app.api.routes.grid_sim import router as grid_sim_router
from app.api.routes.predict import router as predict_router
from app.api.routes.simulation import router as simulation_router
from app.api.routes.tracks import router as tracks_router

app = FastAPI(title="F1 Strategy Simulator")

# ALLOWED_ORIGINS (comma-separated, no trailing slash) lists the deployed frontend origin(s); unset
# keeps the local Next.js dev server only. Set it on Render to the Vercel URL.
_DEFAULT_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000"
ALLOWED_ORIGINS = [
    o.strip().rstrip("/") for o in os.environ.get("ALLOWED_ORIGINS", _DEFAULT_ORIGINS).split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(simulation_router)
app.include_router(grid_sim_router)
app.include_router(tracks_router)
app.include_router(predict_router)
app.include_router(chat_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
