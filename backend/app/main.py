"""
FastAPI app entrypoint.

Run from the backend/ directory:
    uvicorn app.main:app --reload

Or for a quick standalone smoke test from anywhere:
    python backend/app/main.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.simulation import router as simulation_router

app = FastAPI(title="F1 Strategy Simulator")

# local Next.js dev server only -- tighten this before any real deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(simulation_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
