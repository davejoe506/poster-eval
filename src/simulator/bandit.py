"""Contextual bandit policies and reward model for artwork impression simulation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

GENRES = [
    "action",
    "comedy",
    "drama",
    "horror",
    "romance",
    "sci-fi",
    "thriller",
    "documentary",
]

DEVICES = ("mobile", "tv", "web")


@dataclass(frozen=True)
class UserContext:
    user_id: int
    genre_affinity: dict[str, float]
    device_type: str
    account_tenure_days: int

    def to_features(self) -> dict[str, float | str | int]:
        features: dict[str, float | str | int] = {
            "device_type": self.device_type,
            "account_tenure_days": self.account_tenure_days,
        }
        for genre in GENRES:
            features[f"affinity_{genre}"] = self.genre_affinity.get(genre, 0.0)
        return features


@dataclass(frozen=True)
class Title:
    title_id: int
    title_name: str
    genres: tuple[str, ...]
    popularity_score: float
    popularity_tier: str


@dataclass(frozen=True)
class ArtworkVariant:
    artwork_id: int
    title_id: int
    variant_index: int
    genre_emphasis: str  # which genre facet this artwork highlights


@dataclass
class PolicyWeights:
    """Linear weights for scoring artwork variants given context."""

    genre_match: float = 1.0
    variant_quality: float = 0.3
    device_bias: dict[str, float] = field(default_factory=lambda: {"mobile": 0.0, "tv": 0.1, "web": 0.05})
    noise_scale: float = 0.15

    @classmethod
    def logging_default(cls) -> PolicyWeights:
        return cls(genre_match=1.0, variant_quality=0.3)

    @classmethod
    def candidate_v2(cls) -> PolicyWeights:
        """Stronger genre matching — the policy we want to evaluate offline."""
        return cls(genre_match=1.4, variant_quality=0.5, noise_scale=0.1)


def _softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = scores / max(temperature, 1e-6)
    shifted = scaled - scaled.max()
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum()


def score_artwork(
    user: UserContext,
    title: Title,
    artwork: ArtworkVariant,
    weights: PolicyWeights,
) -> float:
    """Compute deterministic utility for an artwork variant."""
    genre_overlap = sum(user.genre_affinity.get(g, 0.0) for g in title.genres)
    emphasis_match = user.genre_affinity.get(artwork.genre_emphasis, 0.0)
    device_bonus = weights.device_bias.get(user.device_type, 0.0)
    variant_bonus = weights.variant_quality * (artwork.variant_index % 3) * 0.1
    return (
        weights.genre_match * (0.6 * genre_overlap + 0.4 * emphasis_match)
        + device_bonus
        + variant_bonus
    )


def action_distribution(
    user: UserContext,
    title: Title,
    artworks: Sequence[ArtworkVariant],
    weights: PolicyWeights,
    *,
    temperature: float = 0.5,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (artwork_ids, probabilities) for all available actions."""
    rng = rng or np.random.default_rng()
    scores = np.array([score_artwork(user, title, a, weights) for a in artworks], dtype=float)
    scores += rng.normal(0, weights.noise_scale, size=len(scores))
    probs = _softmax(scores, temperature=temperature)
    artwork_ids = np.array([a.artwork_id for a in artworks])
    return artwork_ids, probs


def epsilon_greedy_select(
    user: UserContext,
    title: Title,
    artworks: Sequence[ArtworkVariant],
    weights: PolicyWeights,
    epsilon: float,
    rng: np.random.Generator,
    *,
    temperature: float = 0.5,
) -> tuple[int, float, bool]:
    """
    Select an artwork action and return (artwork_id, propensity, is_randomized).

    With probability epsilon, explore uniformly (logged as randomized).
    Otherwise exploit per the softmax policy.
    """
    artwork_ids, probs = action_distribution(
        user, title, artworks, weights, temperature=temperature, rng=rng
    )
    n = len(artwork_ids)
    explore = bool(rng.random() < epsilon)
    if explore:
        idx = int(rng.integers(0, n))
    else:
        idx = int(rng.choice(n, p=probs))

    # Marginal P(a|x) under epsilon-greedy — required for unbiased IPS.
    # Explore/exploit is still flagged via is_randomized for holdout validation.
    mixed_propensity = float(epsilon / n + (1.0 - epsilon) * probs[idx])
    return int(artwork_ids[idx]), mixed_propensity, explore


def sample_reward(
    user: UserContext,
    title: Title,
    artwork: ArtworkVariant,
    rng: np.random.Generator,
    *,
    true_weights: PolicyWeights | None = None,
) -> float:
    """
    Bernoulli reward from a latent logistic model (simulator ground truth).

    Uses slightly different weights than the logging policy so OPE matters.
    """
    tw = true_weights or PolicyWeights(genre_match=1.2, variant_quality=0.45, noise_scale=0.0)
    utility = score_artwork(user, title, artwork, tw)
    popularity_bonus = 0.15 if title.popularity_tier == "head" else 0.0
    logit = 2.2 * utility + popularity_bonus - 1.1
    prob = 1.0 / (1.0 + np.exp(-logit))
    return float(rng.random() < prob)


def policy_probabilities(
    user: UserContext,
    title: Title,
    artworks: Sequence[ArtworkVariant],
    weights: PolicyWeights,
    *,
    temperature: float = 0.5,
) -> dict[int, float]:
    """Deterministic softmax probabilities for a policy (no score noise)."""
    scores = np.array([score_artwork(user, title, a, weights) for a in artworks], dtype=float)
    probs = _softmax(scores, temperature=temperature)
    return {a.artwork_id: float(p) for a, p in zip(artworks, probs)}


def candidate_action_prob(
    user: UserContext,
    title: Title,
    artworks: Sequence[ArtworkVariant],
    artwork_id: int,
    weights: PolicyWeights,
    *,
    temperature: float = 0.5,
) -> float:
    """P(artwork_id | context) under a candidate policy."""
    probs = policy_probabilities(user, title, artworks, weights, temperature=temperature)
    return probs.get(artwork_id, 0.0)


def build_catalog(
    rng: np.random.Generator,
    *,
    n_titles: int = 500,
    variants_per_title: int = 5,
) -> tuple[list[Title], list[ArtworkVariant]]:
    """Generate synthetic title catalog with artwork variants."""
    titles: list[Title] = []
    artworks: list[ArtworkVariant] = []
    popularity_scores = rng.lognormal(mean=0.0, sigma=0.8, size=n_titles)
    tertiles = np.quantile(popularity_scores, [1 / 3, 2 / 3])

    artwork_id = 1
    for i in range(n_titles):
        score = float(popularity_scores[i])
        if score <= tertiles[0]:
            tier = "tail"
        elif score <= tertiles[1]:
            tier = "mid"
        else:
            tier = "head"

        n_genres = int(rng.integers(1, 4))
        genres = tuple(rng.choice(GENRES, size=n_genres, replace=False))
        title = Title(
            title_id=i + 1,
            title_name=f"Title_{i + 1}",
            genres=genres,
            popularity_score=score,
            popularity_tier=tier,
        )
        titles.append(title)

        for v in range(variants_per_title):
            emphasis = str(rng.choice(genres))
            artworks.append(
                ArtworkVariant(
                    artwork_id=artwork_id,
                    title_id=title.title_id,
                    variant_index=v,
                    genre_emphasis=emphasis,
                )
            )
            artwork_id += 1

    return titles, artworks


def build_users(rng: np.random.Generator, *, n_users: int = 1000) -> list[UserContext]:
    """Generate synthetic user contexts with genre affinities."""
    users: list[UserContext] = []
    for user_id in range(1, n_users + 1):
        raw = rng.dirichlet(np.ones(len(GENRES)))
        affinity = {g: float(w) for g, w in zip(GENRES, raw)}
        users.append(
            UserContext(
                user_id=user_id,
                genre_affinity=affinity,
                device_type=str(rng.choice(DEVICES)),
                account_tenure_days=int(rng.integers(1, 2000)),
            )
        )
    return users


def serialize_context(user: UserContext, title: Title) -> str:
    payload = {
        **user.to_features(),
        "title_genres": list(title.genres),
        "popularity_tier": title.popularity_tier,
        "popularity_score": title.popularity_score,
    }
    return json.dumps(payload)
