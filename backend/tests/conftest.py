import sys
from pathlib import Path

# make `app.core.*` / `app.main` importable regardless of the directory
# pytest is invoked from (mirrors the sys.path convention used throughout
# backend/app/core/*.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest


@pytest.fixture
def toy_tire_model() -> dict:
    """
    Small hand-built tire model (NOT fitted from real data) for fast,
    isolated unit tests of race_simulator.py / monte_carlo.py. Degradation
    is a plain linear function per compound (age2_coef=0) so expected
    values can be worked out by hand instead of trusting the code under
    test to check itself.
    """
    return {
        "SOFT": {
            "age_coef": 0.10,
            "age2_coef": 0.0,
            "cliff_age": 10,
            "max_reliable_age": 20,
        },
        "MEDIUM": {
            "age_coef": 0.05,
            "age2_coef": 0.0,
            "cliff_age": 20,
            "max_reliable_age": 30,
        },
        "HARD": {
            "age_coef": 0.02,
            "age2_coef": 0.0,
            "cliff_age": None,
            "max_reliable_age": 40,
        },
        "_fuel_coef": -0.05,
        "_compound_offsets": {"SOFT": -0.5, "MEDIUM": -0.2, "HARD": 0.0},
    }
