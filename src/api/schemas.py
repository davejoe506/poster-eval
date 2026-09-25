"""Pydantic response models for the PosterEval API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    db_available: bool
    service: str = "postereval-api"


class EstimatorValue(BaseModel):
    value: float
    std_error: Optional[float] = None


class PolicyCompareResponse(BaseModel):
    policy_id: str
    eval_timestamp: str
    estimators: Dict[str, EstimatorValue]
    logging_policy_value: Optional[float] = None
    lift_vs_logging: Optional[float] = None
    recommendation: str
    ground_truth_value: Optional[float] = None


class HoldoutResponse(BaseModel):
    candidate_policy: str
    eval_timestamp: str
    snips_absolute_bias: Optional[float] = None
    rmse_across_estimators: Optional[float] = None
    coverage_above_floor: Optional[float] = None
    n_holdout: Optional[int] = None
    n_exploit: Optional[int] = None
    ground_truth_value: Optional[float] = None
    estimators: List[Dict[str, Any]] = Field(default_factory=list)


class StratifiedResponse(BaseModel):
    policy_id: str
    eval_timestamp: str
    tiers: Dict[str, Dict[str, EstimatorValue]]
