"""Tests for OPE estimators."""

import numpy as np
import pytest

from src.ope.estimators import estimate_ips, estimate_snips


def test_ips_snips_known_answer_uniform_policy():
    """When candidate == logging, weights are 1 and SNIPS equals mean reward."""
    rewards = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    logging_p = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    candidate_p = logging_p.copy()

    ips = estimate_ips(rewards, logging_p, candidate_p)
    snips = estimate_snips(rewards, logging_p, candidate_p)

    assert ips.value == pytest.approx(rewards.mean())
    assert snips.value == pytest.approx(rewards.mean())


def test_ips_upweights_rare_action():
    """Rare action with high reward should increase IPS vs naive mean."""
    rewards = np.array([1.0, 0.0])
    logging_p = np.array([0.1, 0.9])  # first action rare
    candidate_p = np.array([0.5, 0.5])  # candidate treats actions equally

    ips = estimate_ips(rewards, logging_p, candidate_p)
    naive = rewards.mean()

    assert ips.value > naive


def test_snips_propensity_floor_prevents_inf():
    """Very small propensities are clipped; estimator returns finite value."""
    rewards = np.array([1.0, 0.0, 1.0])
    logging_p = np.array([1e-6, 0.5, 0.5])
    candidate_p = np.array([0.5, 0.5, 0.5])

    snips = estimate_snips(rewards, logging_p, candidate_p)
    assert np.isfinite(snips.value)
