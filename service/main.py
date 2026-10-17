"""FirstShot API and web UI.

    uvicorn service.main:app --reload --port 8080

OpenAPI docs are served at /docs. The same image runs locally, in Docker and
on AWS Lambda behind the Lambda Web Adapter.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Union

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from firstshot import __version__
from firstshot.engine import get_engine

STATIC = Path(__file__).resolve().parent / "static"

Priority = Literal["fewest", "cost", "carbon", "closest"]


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_engine()          # load models before the first request
    yield


app = FastAPI(
    title="FirstShot",
    version=__version__,
    summary="Paint color matching with fewer pigments, fewer lab trials and safer materials.",
    lifespan=lifespan,
)


# ---------- request schemas ----------
class TargetIn(BaseModel):
    kind: Literal["patch", "hex", "lab", "legacy"] = Field(
        description="patch: ColorChecker patch name. hex: screen color. lab: [L, a, b] under D65. "
                    "legacy: key of a legacy formula.")
    value: Union[str, list[float]] = Field(examples=["orange", "#3a6ea5", [52.0, 10.5, -30.2]])


class MatchIn(BaseModel):
    target: TargetIn
    priority: Priority = "fewest"
    exterior_only: bool = False
    max_colorants: int = Field(3, ge=1, le=3)
    model: Literal["tuned", "hybrid", "physics"] = "tuned"
    mode: Literal["ai", "exhaustive"] = Field("ai", description="ai searches the shortlisted pigment sets; "
                                                                 "exhaustive searches all of them.")


class TrialsIn(BaseModel):
    target: TargetIn
    priority: Priority = "fewest"
    exterior_only: bool = False
    max_colorants: int = Field(3, ge=1, le=3)


class ReformulateIn(BaseModel):
    legacy: str = Field(examples=["safety_yellow"])
    priority: Priority = "cost"
    exterior_only: bool | None = Field(None, description="Defaults to what the legacy formula is used for.")


# ---------- response schemas ----------
class PigmentShare(BaseModel):
    key: str
    name: str
    ci: str
    share: float = Field(description="Percent of the pigment blend by weight")
    chip: str
    status: str


class Recipe(BaseModel):
    pigments: list[PigmentShare]
    colorants: int
    delta_e: dict[str, float] = Field(description="CIEDE2000 to the target under each light")
    pass_probability: float
    passes: bool
    cost_index: float = Field(description="Pigment cost per kg, titanium white = 1")
    carbon_index: float = Field(description="Illustrative embodied-carbon index per kg")
    lab: list[float]
    appearance: dict[str, str]


class CountOption(Recipe):
    chosen: bool


class Option(Recipe):
    priority: Priority
    viable: bool
    cost_pct: float
    carbon_pct: float


class TargetOut(BaseModel):
    name: str
    spectral: bool
    lab: list[float]
    appearance: dict[str, str]


class Search(BaseModel):
    mode: str
    sets_checked: int
    sets_total: int
    recipes_scored: int
    ms: int
    widened: bool = Field(description="True when the shortlist found nothing that passes and the search widened")


class VersusClosest(BaseModel):
    same_recipe: bool
    pigments_saved: int
    cost_pct: float
    carbon_pct: float


class MatchOut(BaseModel):
    target: TargetOut
    viable: bool
    recipe: Recipe
    closest: Recipe
    by_count: list[CountOption]
    versus_closest: VersusClosest
    search: Search
    settings: dict


class TrialStep(BaseModel):
    trial: int
    de_daylight: float
    de_other_lights: float | None
    passed: bool


class TrialRun(BaseModel):
    model: str
    label: str
    pigments: int
    trials: int
    passed: bool
    history: list[TrialStep]


class TrialsOut(BaseModel):
    target: TargetOut
    runs: list[TrialRun]


class LegacyFormula(BaseModel):
    key: str
    name: str
    use: str
    exterior: bool
    reason: str
    pigments: list[PigmentShare]
    replaced: list[str]
    cost_index: float
    carbon_index: float


class ReformulateOut(BaseModel):
    legacy: LegacyFormula
    target: TargetOut
    viable: bool
    replacement: Recipe
    change: dict[str, float]
    lab_check: TrialRun
    options: list[Option]
    settings: dict


class LegacySummary(BaseModel):
    key: str
    name: str
    use: str
    exterior: bool
    reason: str
    hex: str
    pigments: list[PigmentShare]


class PigmentInfo(BaseModel):
    key: str
    name: str
    ci: str
    status: str
    exterior: bool
    organic: bool
    cost: float
    carbon: float
    note: str
    chip: str
    masstone: str


class PatchInfo(BaseModel):
    name: str
    hex: str


# ---------- routes ----------
def _run(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/health", tags=["service"])
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/api/palette", response_model=list[PigmentInfo], tags=["reference"])
def palette():
    return get_engine().palette()


@app.get("/api/targets", response_model=list[PatchInfo], tags=["reference"])
def targets():
    return get_engine().targets()


@app.get("/api/legacy", response_model=list[LegacySummary], tags=["reference"])
def legacy():
    return get_engine().legacy()


@app.get("/api/model-card", tags=["reference"])
def model_card() -> dict:
    return get_engine().model_card()


@app.post("/api/match", response_model=MatchOut, tags=["formulation"])
def match(req: MatchIn):
    """Find the recipe with the fewest pigments that passes under four lights."""
    return _run(get_engine().match, req.target.kind, req.target.value, priority=req.priority,
                exterior_only=req.exterior_only, max_colorants=req.max_colorants, model_key=req.model,
                mode=req.mode)


@app.post("/api/trials", response_model=TrialsOut, tags=["formulation"])
def trials(req: TrialsIn):
    """Simulate spray, measure and correct until the color passes, for each model."""
    return _run(get_engine().trials, req.target.kind, req.target.value, priority=req.priority,
                exterior_only=req.exterior_only, max_colorants=req.max_colorants)


@app.post("/api/reformulate", response_model=ReformulateOut, tags=["materials"])
def reformulate(req: ReformulateIn):
    """Rebuild a legacy color without its restricted or costly pigment."""
    return _run(get_engine().reformulate, req.legacy, priority=req.priority, exterior_only=req.exterior_only)


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")
