"""Bias, RMSE, and coverage metrics for OPE vs. ground truth."""

from __future__ import annotations

import numpy as np

from src.ope.estimators import MIN_PROPENSITY


def absolute_bias(estimate: float, ground_truth: float) -> float:
    """Signed error is useful too, but absolute bias is the spec success metric."""
    return float(abs(estimate - ground_truth))


def signed_error(estimate: float, ground_truth: float) -> float:
    return float(estimate - ground_truth)


def rmse(estimates: np.ndarray, ground_truths: np.ndarray) -> float:
    estimates = np.asarray(estimates, dtype=float)
    ground_truths = np.asarray(ground_truths, dtype=float)
    if estimates.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((estimates - ground_truths) ** 2)))


def coverage_above_floor(propensities: np.ndarray, floor: float = MIN_PROPENSITY) -> float:
    """Share of impressions with propensity above the IPS clip floor."""
    p = np.asarray(propensities, dtype=float)
    if p.size == 0:
        return 0.0
    return float((p > floor).mean())


def estimation_error(estimate: float, ground_truth: float) -> dict[str, float]:
    return {
        "ground_truth": float(ground_truth),
        "estimate": float(estimate),
        "signed_error": signed_error(estimate, ground_truth),
        "absolute_bias": absolute_bias(estimate, ground_truth),
    }
