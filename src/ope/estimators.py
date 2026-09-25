"""Off-policy evaluation estimators: IPS, SNIPS, Doubly Robust."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


MIN_PROPENSITY = 1e-3


@dataclass
class OPEResult:
    estimator: str
    value: float
    std_error: float | None = None
    n_samples: int = 0
    effective_sample_size: float | None = None

    def to_dict(self) -> dict:
        return {
            "estimator": self.estimator,
            "value": self.value,
            "std_error": self.std_error,
            "n_samples": self.n_samples,
            "effective_sample_size": self.effective_sample_size,
        }


def _clip_propensity(p: np.ndarray, floor: float = MIN_PROPENSITY) -> np.ndarray:
    return np.clip(p, floor, 1.0)


def importance_weights(
    logging_propensity: np.ndarray,
    candidate_propensity: np.ndarray,
    clip: float = 100.0,
) -> np.ndarray:
    """Compute w = pi_b(a|x) / pi_e(a|x) with propensity floor and weight clip."""
    pi_e = _clip_propensity(logging_propensity)
    pi_b = np.clip(candidate_propensity, 0.0, 1.0)
    weights = pi_b / pi_e
    return np.clip(weights, 0.0, clip)


def estimate_ips(
    rewards: np.ndarray,
    logging_propensity: np.ndarray,
    candidate_propensity: np.ndarray,
) -> OPEResult:
    """
    Inverse Propensity Scoring (IPS) policy value estimator.

    V_IPS = (1/n) * sum_i [ r_i * pi_b(a_i|x_i) / pi_e(a_i|x_i) ]
    """
    rewards = np.asarray(rewards, dtype=float)
    weights = importance_weights(logging_propensity, candidate_propensity)
    n = len(rewards)
    if n == 0:
        return OPEResult(estimator="IPS", value=0.0, n_samples=0)

    weighted_rewards = weights * rewards
    value = float(weighted_rewards.mean())
    # Delta-method style SE for IPS
    var = float(weighted_rewards.var(ddof=1) / n) if n > 1 else 0.0
    ess = float((weights.sum() ** 2) / (weights**2).sum()) if (weights**2).sum() > 0 else 0.0

    return OPEResult(
        estimator="IPS",
        value=value,
        std_error=float(np.sqrt(var)) if var >= 0 else None,
        n_samples=n,
        effective_sample_size=ess,
    )


def estimate_snips(
    rewards: np.ndarray,
    logging_propensity: np.ndarray,
    candidate_propensity: np.ndarray,
) -> OPEResult:
    """
    Self-Normalized IPS (SNIPS) — primary reporting metric.

    V_SNIPS = sum(w_i * r_i) / sum(w_i)
    """
    rewards = np.asarray(rewards, dtype=float)
    weights = importance_weights(logging_propensity, candidate_propensity)
    n = len(rewards)
    if n == 0 or weights.sum() == 0:
        return OPEResult(estimator="SNIPS", value=0.0, n_samples=n)

    value = float((weights * rewards).sum() / weights.sum())
    ess = float((weights.sum() ** 2) / (weights**2).sum())

    return OPEResult(
        estimator="SNIPS",
        value=value,
        n_samples=n,
        effective_sample_size=ess,
    )


def estimate_doubly_robust(
    rewards: np.ndarray,
    logging_propensity: np.ndarray,
    candidate_propensity: np.ndarray,
    reward_predictions: np.ndarray,
    candidate_probs_matrix: np.ndarray | None = None,
) -> OPEResult:
    """
    Doubly Robust (DR) estimator using logged-action form.

    When candidate_probs_matrix is None, uses simplified DR:
        V_DR = (1/n) * sum [ w_i * (r_i - mu_i) + mu_i ]

    With full matrix (n x n_actions), uses:
        V_DR = (1/n) * sum_i [ w_i * (r_i - mu(x_i,a_i)) + sum_a pi_b(a|x_i) * mu(x_i,a) ]
    """
    rewards = np.asarray(rewards, dtype=float)
    mu = np.asarray(reward_predictions, dtype=float)
    weights = importance_weights(logging_propensity, candidate_propensity)
    n = len(rewards)
    if n == 0:
        return OPEResult(estimator="DR", value=0.0, n_samples=0)

    correction = weights * (rewards - mu)
    if candidate_probs_matrix is not None:
        direct = (candidate_probs_matrix * mu.reshape(-1, 1)).sum(axis=1) if mu.ndim == 1 else (candidate_probs_matrix * mu).sum(axis=1)
        dr_terms = correction + direct
    else:
        dr_terms = correction + mu

    value = float(dr_terms.mean())
    var = float(dr_terms.var(ddof=1) / n) if n > 1 else 0.0
    ess = float((weights.sum() ** 2) / (weights**2).sum()) if (weights**2).sum() > 0 else 0.0

    return OPEResult(
        estimator="DR",
        value=value,
        std_error=float(np.sqrt(var)) if var >= 0 else None,
        n_samples=n,
        effective_sample_size=ess,
    )


def run_all_estimators(
    logs: pd.DataFrame,
    candidate_propensity: np.ndarray,
    reward_predictions: np.ndarray | None = None,
) -> list[OPEResult]:
    """Run IPS, SNIPS, and optionally DR on a log dataframe."""
    rewards = logs["reward"].to_numpy()
    logging_p = logs["logging_propensity"].to_numpy()

    results = [
        estimate_ips(rewards, logging_p, candidate_propensity),
        estimate_snips(rewards, logging_p, candidate_propensity),
    ]

    if reward_predictions is not None:
        results.append(
            estimate_doubly_robust(rewards, logging_p, candidate_propensity, reward_predictions)
        )

    return results
