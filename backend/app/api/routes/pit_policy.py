"""
POST /strategy/policy (T-055): the best safety-car-reactive pit rule for one car's base plan, scored with the
full-grid simulation. Thin wrapper over app.core.pit_policy; read its docstring for the model and the search.
"""

import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.api.deps import resolve_track
from app.api.routes.simulation import MAX_RACE_LAPS, StintIn, get_tire_model
from app.core import pit_policy as PP
from app.core.race_simulator import Stint

router = APIRouter()


class PolicyRequest(BaseModel):
    track_id: str = Field(description="Track from GET /tracks")
    driver: Optional[str] = Field(default=None, description="Three-letter code of the car the rule is for. Default: the driver on pole.")
    round: Optional[int] = Field(default=None, ge=1, le=30, description="2026 round for the field. Default: the round at track_id.")
    base_plan: Optional[List[StintIn]] = Field(
        default=None, description="The fixed plan the rule is added to (must sum to the race laps). Default: the optimizer's #1 plan."
    )
    n_sims: int = Field(default=3000, ge=200, le=10_000, description="Runs for the reported (fresh-seed) gain")
    n_screen: int = Field(default=600, ge=100, le=3_000, description="Runs per candidate during the search")
    seed: int = Field(default=0, ge=0)
    safety_cars: bool = Field(default=True, description="False = no random SC/VSC: the rule can never fire and the gain is 0")
    field_sensitivity: bool = Field(default=True, description="Also score the rule with every other car following it too")
    total_laps: Optional[int] = Field(default=None, gt=0, le=MAX_RACE_LAPS)


class PolicyOut(BaseModel):
    window_lo: int = Field(description="First lap an SC/VSC may start on for the rule to apply")
    window_hi: int
    min_age: int = Field(description="Minimum tyre age (laps) on the lap the SC starts")
    compound: str = Field(description="SOFT | MEDIUM | HARD: the tyre to box for")
    trigger: str = Field(description="'sc' = full safety car only, 'sc_or_vsc'")


class StintOut(BaseModel):
    compound: str
    laps: int
    start_age: int


class GainOut(BaseModel):
    points: float = Field(description="Expected championship points gained vs the fixed plan, all runs")
    points_ci95: float = Field(description="Half-width of the 95% interval on points (paired runs)")
    positions: float = Field(description="Expected finishing positions gained (+ = better)")
    positions_ci95: float
    fire_rate: float = Field(description="Share of runs in which the rule fired")


class PolicyResponse(BaseModel):
    track_id: str
    round: int
    total_laps: int
    driver: str
    team: str
    grid: int
    base_plan: List[StintOut]
    policy: Optional[PolicyOut]
    rule: str = Field(description="The rule in plain English, with a caveat when it is not reliably better")
    significant: bool = Field(description="The 95% interval of the points gain excludes 0 (positive side)")
    gain: Optional[GainOut]
    points_gain_when_fired: Optional[float]
    points_gain_when_fired_ci95: Optional[float]
    positions_gain_when_fired: Optional[float]
    share_of_fires_better: Optional[float] = Field(description="Among runs where it fired: share with more points than the fixed plan")
    share_of_fires_worse: Optional[float]
    fire_lap_mean: Optional[float]
    base_exp_points: Optional[float] = Field(description="Expected points of the fixed plan (same runs)")
    base_mean_position: Optional[float]
    if_field_follows: Optional[GainOut] = Field(
        description="Same comparison when every other car also follows the rule (the rule vs the fixed plan, field reacting)"
    )
    screen_gain_points: float = Field(description="Gain on the search sample (optimistic: it was selected on it)")
    n_sims: int
    n_screen: int
    seed: int
    notes: List[str]
    assumptions: List[str]


ASSUMPTIONS = [
    "Scored with the full-grid simulation (traffic, undercut, SC bunching, the field's own fixed-plan stops); every other car "
    "keeps a fixed plan unless 'if_field_follows' is shown.",
    "Safety-car and VSC timing, length and cheaper pit stops are the fitted values from ~40 races (26 SC and 30 VSC events); "
    "a policy cannot be tested on events the data has few of (early SCs, late VSCs).",
    "The stop is pulled forward: the next planned stop moves to the SC lap on the chosen compound (no extra stop) unless the car "
    "is on its last stint, where it is an extra stop. New stints longer than the compound's observed cap are not allowed.",
    "Tyre wear and fuel burn are not reduced under SC/VSC, the new tyre goes on at the start of the SC lap, and the tyre "
    "cliff penalties are placeholders: how much a fresh set is worth late in a stint rests on those.",
    "Rule chosen on one random seed and scored on a different one (the interval does not include the search's selection).",
]


def _gain(ev: PP.PolicyEvaluation) -> GainOut:
    return GainOut(
        points=ev.gain_pts, points_ci95=ev.gain_pts_ci, positions=ev.gain_pos, positions_ci95=ev.gain_pos_ci,
        fire_rate=ev.fire_rate,
    )


@router.post("/strategy/policy", response_model=PolicyResponse)
def policy_route(request: PolicyRequest) -> PolicyResponse:
    track = resolve_track(request.track_id)
    tire_model = get_tire_model()
    plan = [Stint(s.compound, s.laps, s.start_age) for s in request.base_plan] if request.base_plan else None
    try:
        ctx = PP.build_context(
            request.track_id, tire_model, driver=request.driver, base_plan=plan, rnd=request.round,
            safety_cars=request.safety_cars, seed=request.seed, total_laps=request.total_laps,
        )
        res = PP.search_policy(
            ctx, n_screen=request.n_screen, n_eval=request.n_sims, seed=request.seed,
            field_sensitivity=request.field_sensitivity,
        )
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    car = ctx.cars[ctx.hero]
    ev, pol = res.evaluation, res.best
    notes = list(res.notes)
    if not request.safety_cars:
        notes.append("safety_cars is false: there are no SC/VSC events, so no rule can fire")
    return PolicyResponse(
        track_id=track["track_id"],
        round=_round_of(request, ctx),
        total_laps=ctx.total_laps,
        driver=car.code,
        team=car.team,
        grid=car.grid,
        base_plan=[StintOut(compound=s.compound, laps=s.laps, start_age=s.start_age) for s in car.strategy],
        policy=PolicyOut(window_lo=pol.window_lo, window_hi=pol.window_hi, min_age=pol.min_age, compound=pol.compound, trigger=pol.trigger) if pol else None,
        rule=res.rule,
        significant=res.significant,
        gain=_gain(ev) if ev else None,
        points_gain_when_fired=ev.gain_pts_when_fired if ev else None,
        points_gain_when_fired_ci95=ev.gain_pts_when_fired_ci if ev else None,
        positions_gain_when_fired=ev.gain_pos_when_fired if ev else None,
        share_of_fires_better=ev.p_gain if ev else None,
        share_of_fires_worse=ev.p_loss if ev else None,
        fire_lap_mean=ev.fire_lap_mean if ev else None,
        base_exp_points=ev.base_exp_points if ev else None,
        base_mean_position=ev.base_mean_position if ev else None,
        if_field_follows=_gain(res.field_evaluation) if res.field_evaluation else None,
        screen_gain_points=res.screen_gain_pts,
        n_sims=request.n_sims,
        n_screen=request.n_screen,
        seed=request.seed,
        notes=notes,
        assumptions=ASSUMPTIONS,
    )


def _round_of(request: PolicyRequest, ctx: PP.PolicyContext) -> int:
    from app.core import grid_inputs as G

    return int(request.round or G.round_for_track(G.load_history(), request.track_id) or 0)
