"""Tests for holdout validation and error metrics."""

import numpy as np
import pandas as pd
import pytest

from src.ope.estimators import OPEResult
from src.reporting.metrics import absolute_bias, coverage_above_floor, rmse
from src.simulator.bandit import PolicyWeights
from src.simulator.generate_logs import generate_bandit_logs
from src.validation.holdout import (
    compare_ope_to_holdout,
    holdout_unbiased_value,
    propensity_stratified_ate,
    validate_policy,
)


def test_absolute_bias_and_rmse():
    assert absolute_bias(0.14, 0.12) == pytest.approx(0.02)
    assert rmse(np.array([0.1, 0.2]), np.array([0.1, 0.0])) == pytest.approx(np.sqrt(0.02))


def test_coverage_above_floor():
    p = np.array([0.5, 0.002, 1e-4])
    assert coverage_above_floor(p, floor=1e-3) == pytest.approx(2 / 3)


def test_holdout_ips_known_answer_uniform_candidate():
    logs = pd.DataFrame(
        {
            "reward": [1.0, 0.0, 1.0, 0.0],
            "variant_index": [0, 1, 0, 1],
            "is_randomized": [True, True, True, True],
        }
    )
    candidate_p = np.array([0.5, 0.5, 0.5, 0.5])
    result = holdout_unbiased_value(logs, candidate_p, n_actions=2)
    assert result.value == pytest.approx(0.5)


def test_holdout_ips_upweights_preferred_action():
    logs = pd.DataFrame(
        {
            "reward": [1.0, 0.0, 1.0, 0.0],
            "variant_index": [0, 1, 0, 1],
            "is_randomized": [True, True, True, True],
        }
    )
    # Candidate always wants action 0; holdout logged 0 with reward 1.
    candidate_p = np.array([1.0, 0.0, 1.0, 0.0])
    result = holdout_unbiased_value(logs, candidate_p, n_actions=2)
    assert result.value == pytest.approx(1.0)


def test_compare_ope_records_bias():
    results = [OPEResult(estimator="SNIPS", value=0.15, n_samples=10)]
    compared = compare_ope_to_holdout(results, ground_truth=0.12)
    assert compared[0]["absolute_bias"] == pytest.approx(0.03)
    assert compared[0]["signed_error"] == pytest.approx(0.03)


def test_propensity_stratified_ate_recovers_constant_gap():
    rewards = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    treatment = np.array([True, True, True, False, False, False])
    propensity = np.array([0.1, 0.5, 0.9, 0.1, 0.5, 0.9])
    out = propensity_stratified_ate(rewards, treatment, propensity, n_bins=3)
    assert out["ate"] == pytest.approx(1.0)
    assert out["n_bins_used"] == 3


def test_validate_policy_on_simulated_logs():
    from src.ope.evaluate import compute_candidate_propensities

    logs = generate_bandit_logs(n_impressions=3000, n_titles=30, n_users=60, seed=2)
    candidate_p = compute_candidate_propensities(logs, PolicyWeights.candidate_v2(), seed=2)
    report = validate_policy(logs, candidate_p)

    assert report["n_holdout"] > 0
    assert report["n_exploit"] > 0
    assert report["snips_absolute_bias"] is not None
    assert np.isfinite(report["snips_absolute_bias"])
    assert "head" in report["stratified_by_tier"] or "mid" in report["stratified_by_tier"]
    assert "ate" in report["propensity_score_matching"]
