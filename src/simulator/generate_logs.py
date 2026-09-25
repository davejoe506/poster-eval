"""CLI to generate synthetic bandit impression logs."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.simulator.bandit import (
    build_catalog,
    build_users,
    epsilon_greedy_select,
    sample_reward,
    serialize_context,
)


def generate_bandit_logs(
    *,
    n_impressions: int = 20_000,
    n_titles: int = 200,
    n_users: int = 500,
    variants_per_title: int = 5,
    epsilon: float = 0.10,
    seed: int = 42,
    logging_policy: str = "logging_v1",
) -> pd.DataFrame:
    """Simulate contextual bandit impressions with logged propensities."""
    rng = np.random.default_rng(seed)
    from src.simulator.bandit import PolicyWeights

    logging_weights = PolicyWeights.logging_default()

    titles, artworks = build_catalog(
        rng, n_titles=n_titles, variants_per_title=variants_per_title
    )
    users = build_users(rng, n_users=n_users)

    artworks_by_title: dict[int, list] = defaultdict(list)
    for artwork in artworks:
        artworks_by_title[artwork.title_id].append(artwork)

    title_lookup = {t.title_id: t for t in titles}
    user_lookup = {u.user_id: u for u in users}

    base_time = datetime(2026, 1, 1)
    records: list[dict] = []

    for impression_id in range(1, n_impressions + 1):
        user = users[int(rng.integers(0, len(users)))]
        title = titles[int(rng.integers(0, len(titles)))]
        available = artworks_by_title[title.title_id]

        artwork_id, propensity, is_randomized = epsilon_greedy_select(
            user,
            title,
            available,
            logging_weights,
            epsilon=epsilon,
            rng=rng,
        )

        artwork = next(a for a in available if a.artwork_id == artwork_id)
        reward = sample_reward(user, title, artwork, rng)

        records.append(
            {
                "impression_id": impression_id,
                "timestamp": base_time + timedelta(seconds=int(rng.integers(0, 86400 * 30))),
                "user_id": user.user_id,
                "title_id": title.title_id,
                "artwork_id": artwork_id,
                "variant_index": artwork.variant_index,
                "genre_emphasis": artwork.genre_emphasis,
                "context_features": serialize_context(user, title),
                "logging_propensity": propensity,
                "reward": reward,
                "logging_policy": logging_policy,
                "is_randomized": is_randomized,
                "popularity_tier": title.popularity_tier,
            }
        )

    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic bandit impression logs")
    parser.add_argument("--output", type=Path, default=Path("data/processed/bandit_logs.parquet"))
    parser.add_argument("--n-impressions", type=int, default=20_000)
    parser.add_argument("--n-titles", type=int, default=200)
    parser.add_argument("--n-users", type=int, default=500)
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    df = generate_bandit_logs(
        n_impressions=args.n_impressions,
        n_titles=args.n_titles,
        n_users=args.n_users,
        epsilon=args.epsilon,
        seed=args.seed,
    )

    df.to_parquet(args.output, index=False)
    print(f"Wrote {len(df):,} impressions to {args.output}")
    print(f"  Mean reward: {df['reward'].mean():.4f}")
    print(f"  Randomized share: {df['is_randomized'].mean():.2%}")
    print(f"  Min propensity: {df['logging_propensity'].min():.6f}")


if __name__ == "__main__":
    main()
