"""FastAPI service exposing PosterEval OPE results from SQLite."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query

from src.api.schemas import (
    HealthResponse,
    HoldoutResponse,
    PolicyCompareResponse,
    StratifiedResponse,
)
from src.api.store import (
    DEFAULT_CANDIDATE,
    db_available,
    fetch_holdout,
    fetch_policy_compare,
    fetch_stratified,
)

app = FastAPI(
    title="PosterEval API",
    description="Off-policy evaluation results for streaming artwork personalization.",
    version="0.1.0",
)


def _http_lookup(exc: Exception) -> HTTPException:
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=404, detail=str(exc) + " Run the Airflow DAG first.")
    if isinstance(exc, LookupError):
        return HTTPException(status_code=404, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", db_available=db_available())


@app.get("/policies/compare", response_model=PolicyCompareResponse)
def compare_policies(
    policy_id: str = Query(default=DEFAULT_CANDIDATE, description="Candidate policy id"),
) -> PolicyCompareResponse:
    try:
        return PolicyCompareResponse.model_validate(fetch_policy_compare(policy_id))
    except (FileNotFoundError, LookupError) as exc:
        raise _http_lookup(exc) from exc


@app.get("/validation/holdout", response_model=HoldoutResponse)
def holdout_validation(
    policy_id: str = Query(default=DEFAULT_CANDIDATE, description="Candidate policy id"),
) -> HoldoutResponse:
    try:
        return HoldoutResponse.model_validate(fetch_holdout(policy_id))
    except (FileNotFoundError, LookupError) as exc:
        raise _http_lookup(exc) from exc


@app.get("/reports/stratified", response_model=StratifiedResponse)
def stratified_report(
    policy_id: str = Query(default=DEFAULT_CANDIDATE, description="Candidate policy id"),
) -> StratifiedResponse:
    try:
        return StratifiedResponse.model_validate(fetch_stratified(policy_id))
    except (FileNotFoundError, LookupError) as exc:
        raise _http_lookup(exc) from exc
