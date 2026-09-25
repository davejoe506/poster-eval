"""Tests for bandit simulator."""

import numpy as np
import pytest

from src.simulator.bandit import (
    PolicyWeights,
    build_catalog,
    build_users,
    epsilon_greedy_select,
)
from src.simulator.generate_logs import generate_bandit_logs


def test_epsilon_greedy_propensities_sum_for_explore():
    rng = np.random.default_rng(0)
    titles, artworks = build_catalog(rng, n_titles=5, variants_per_title=4)
    users = build_users(rng, n_users=3)
    by_title = {}
    for a in artworks:
        by_title.setdefault(a.title_id, []).append(a)

    user = users[0]
    title = titles[0]
    available = by_title[title.title_id]

    _, propensity, is_randomized = epsilon_greedy_select(
        user, title, available, PolicyWeights.logging_default(), epsilon=1.0, rng=rng
    )
    assert is_randomized is True
    assert propensity == pytest.approx(1.0 / len(available))


def test_epsilon_greedy_logs_mixture_propensity():
    """Logged propensity is P(a|x) = ε/K + (1-ε) π_softmax, not the branch probability."""
    rng = np.random.default_rng(1)
    titles, artworks = build_catalog(rng, n_titles=5, variants_per_title=4)
    users = build_users(rng, n_users=3)
    by_title: dict[int, list] = {}
    for a in artworks:
        by_title.setdefault(a.title_id, []).append(a)

    available = by_title[titles[0].title_id]
    k = len(available)
    epsilon = 0.10
    _, propensity, _ = epsilon_greedy_select(
        users[0],
        titles[0],
        available,
        PolicyWeights.logging_default(),
        epsilon=epsilon,
        rng=rng,
    )
    # Mixture propensity is bounded below by ε/K and above by ε/K + (1-ε).
    assert propensity >= epsilon / k - 1e-12
    assert propensity <= epsilon / k + (1.0 - epsilon) + 1e-12


def test_generate_logs_schema():
    df = generate_bandit_logs(n_impressions=500, n_titles=20, n_users=50, seed=1)
    required = {
        "impression_id",
        "logging_propensity",
        "reward",
        "artwork_id",
        "variant_index",
        "genre_emphasis",
        "is_randomized",
        "popularity_tier",
    }
    assert required.issubset(df.columns)
    assert (df["logging_propensity"] > 0).all()
    assert df["reward"].isin([0, 1]).all()
    assert df["is_randomized"].mean() > 0.05  # ~10% epsilon
