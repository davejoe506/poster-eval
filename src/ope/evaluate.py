"""Policy comparison runner for off-policy evaluation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.ope.estimators import run_all_estimators
from src.propensity.train import fit_reward_model, train_propensity_model
from src.simulator.bandit import (
    ArtworkVariant,
    PolicyWeights,
    build_catalog,
    build_users,
    candidate_action_prob,
)
from src.validation.holdout import n_actions_from_logs, validate_policy


def _rebuild_catalog_from_logs(logs: pd.DataFrame, seed: int = 42):
    """Rebuild catalog structures matching log generation (same seed params)."""
    rng = np.random.default_rng(seed)
    title_ids = logs["title_id"].unique()
    user_ids = logs["user_id"].unique()
    n_titles = max(int(title_ids.max()), len(title_ids))
    n_users = max(int(user_ids.max()), len(user_ids))

    titles, artworks = build_catalog(rng, n_titles=n_titles, variants_per_title=5)
    users = build_users(rng, n_users=n_users)

    title_lookup = {t.title_id: t for t in titles}
    user_lookup = {u.user_id: u for u in users}
    artworks_by_title: dict[int, list[ArtworkVariant]] = defaultdict(list)
    for a in artworks:
        artworks_by_title[a.title_id].append(a)

    return title_lookup, user_lookup, artworks_by_title


def compute_candidate_propensities(
    logs: pd.DataFrame,
    candidate_weights: PolicyWeights,
    *,
    seed: int = 42,
) -> np.ndarray:
    """pi_b(a|x) for each logged (context, action) pair under the candidate policy."""
    title_lookup, user_lookup, artworks_by_title = _rebuild_catalog_from_logs(logs, seed=seed)
    probs: list[float] = []

    for row in logs.itertuples():
        user = user_lookup.get(row.user_id)
        title = title_lookup.get(row.title_id)
        if user is None or title is None:
            probs.append(0.0)
            continue
        available = artworks_by_title.get(row.title_id, [])
        if not available:
            probs.append(0.0)
            continue
        p = candidate_action_prob(user, title, available, row.artwork_id, candidate_weights)
        probs.append(p)

    return np.array(probs, dtype=float)


def evaluate_policies(
    logs: pd.DataFrame,
    candidate_weights: PolicyWeights | None = None,
    *,
    seed: int = 42,
    propensity_model_type: str = "xgboost",
) -> dict:
    """Run OPE for logging baseline vs candidate policy; return structured report."""
    candidate_weights = candidate_weights or PolicyWeights.candidate_v2()

    candidate_p = compute_candidate_propensities(logs, candidate_weights, seed=seed)
    mu = fit_reward_model(logs, seed=seed)

    propensity = train_propensity_model(logs, model_type=propensity_model_type, seed=seed)
    estimated_p = propensity.predict_logged_propensity(logs)
    oracle_p = logs["logging_propensity"].to_numpy(dtype=float)

    candidate_oracle = run_all_estimators(logs, candidate_p, reward_predictions=mu)

    estimated_logs = logs.copy()
    estimated_logs["logging_propensity"] = estimated_p
    candidate_estimated = run_all_estimators(estimated_logs, candidate_p, reward_predictions=mu)

    # Logging policy evaluated against itself (pi_b == pi_e -> unit weights)
    logging_results = run_all_estimators(logs, oracle_p, reward_predictions=mu)

    snips_candidate = next(r for r in candidate_oracle if r.estimator == "SNIPS")
    snips_logging = next(r for r in logging_results if r.estimator == "SNIPS")

    k = n_actions_from_logs(logs)
    validation_oracle = validate_policy(
        logs, candidate_p, logging_propensity=oracle_p, reward_predictions=mu, n_actions=k
    )
    validation_estimated = validate_policy(
        logs, candidate_p, logging_propensity=estimated_p, reward_predictions=mu, n_actions=k
    )

    stratified: dict[str, list[dict]] = {}
    for tier in ("head", "mid", "tail"):
        tier_mask = (logs["popularity_tier"] == tier).to_numpy()
        if not tier_mask.any():
            continue
        tier_results = run_all_estimators(
            logs.loc[tier_mask],
            candidate_p[tier_mask],
            reward_predictions=mu[tier_mask],
        )
        stratified[tier] = [r.to_dict() for r in tier_results]

    return {
        "logging_policy": {
            "snips_value": snips_logging.value,
            "estimators": [r.to_dict() for r in logging_results],
        },
        "candidate_policy": "bandit_v2",
        "candidate_results_oracle_propensity": [r.to_dict() for r in candidate_oracle],
        "candidate_results_estimated_propensity": [r.to_dict() for r in candidate_estimated],
        "lift_vs_logging_snips": snips_candidate.value - snips_logging.value,
        "propensity_model": {
            "model_type": propensity.model_type,
            "feature_names": propensity.feature_names,
            "n_classes": propensity.n_classes,
            "metrics": propensity.metrics,
        },
        "validation_oracle_propensity": validation_oracle,
        "validation_estimated_propensity": validation_estimated,
        "stratified_by_tier": stratified,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run off-policy evaluation on bandit logs")
    parser.add_argument("--logs", type=Path, default=Path("data/processed/bandit_logs.parquet"))
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--propensity-model",
        choices=("xgboost", "logistic"),
        default="xgboost",
        help="Estimator for P(action | context)",
    )
    args = parser.parse_args()

    logs = pd.read_parquet(args.logs)
    report = evaluate_policies(logs, seed=args.seed, propensity_model_type=args.propensity_model)

    print(json.dumps(report, indent=2))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote report to {args.output}")


if __name__ == "__main__":
    main()
