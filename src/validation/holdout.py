"""Causal validation: randomized holdout comparison and propensity matching."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.ope.estimators import OPEResult, estimate_ips, run_all_estimators
from src.reporting.metrics import estimation_error, rmse


def randomized_subset(logs: pd.DataFrame) -> pd.DataFrame:
    return logs.loc[logs["is_randomized"]].copy()


def exploited_subset(logs: pd.DataFrame) -> pd.DataFrame:
    return logs.loc[~logs["is_randomized"]].copy()


def n_actions_from_logs(logs: pd.DataFrame) -> int:
    if "variant_index" not in logs.columns or logs.empty:
        return 5
    return int(logs["variant_index"].max()) + 1


def holdout_unbiased_value(
    randomized_logs: pd.DataFrame,
    candidate_propensity: np.ndarray,
    *,
    n_actions: int | None = None,
) -> OPEResult:
    """
    Unbiased V(π_b) on the uniform-random holdout.

    Conditional on is_randomized, actions were drawn uniformly, so
    π_e(a|x) = 1/K and IPS is a ground-truth proxy for the candidate policy.
    """
    if randomized_logs.empty:
        return OPEResult(estimator="holdout_IPS", value=float("nan"), n_samples=0)

    k = n_actions if n_actions is not None else n_actions_from_logs(randomized_logs)
    uniform_p = np.full(len(randomized_logs), 1.0 / k)
    result = estimate_ips(
        randomized_logs["reward"].to_numpy(dtype=float),
        uniform_p,
        np.asarray(candidate_propensity, dtype=float),
    )
    result.estimator = "holdout_IPS"
    return result


def compare_ope_to_holdout(
    ope_results: list[OPEResult],
    ground_truth: float,
) -> list[dict]:
    compared: list[dict] = []
    for result in ope_results:
        row = result.to_dict()
        row.update(estimation_error(result.value, ground_truth))
        compared.append(row)
    return compared


def propensity_stratified_ate(
    rewards: np.ndarray,
    treatment: np.ndarray,
    propensity: np.ndarray,
    *,
    n_bins: int = 5,
) -> dict:
    """
    Propensity-score stratified ATE.

    Treatment = candidate policy prefers the logged action more than uniform
    (π_b > 1/K). Within propensity quintiles, compare mean reward treated vs control.
    """
    rewards = np.asarray(rewards, dtype=float)
    treatment = np.asarray(treatment, dtype=bool)
    propensity = np.asarray(propensity, dtype=float)

    if len(rewards) == 0 or treatment.sum() == 0 or (~treatment).sum() == 0:
        return {"ate": float("nan"), "n_matched": 0, "n_bins_used": 0}

    try:
        bins = pd.Series(pd.qcut(propensity, n_bins, duplicates="drop"))
    except ValueError:
        return {"ate": float("nan"), "n_matched": 0, "n_bins_used": 0}

    weighted_ates: list[tuple[float, int]] = []
    for category in bins.cat.categories:
        mask_np = (bins == category).to_numpy()
        t = treatment[mask_np]
        r = rewards[mask_np]
        n_t = int(t.sum())
        n_c = int((~t).sum())
        if n_t == 0 or n_c == 0:
            continue
        ate = float(r[t].mean() - r[~t].mean())
        weighted_ates.append((ate, int(mask_np.sum())))

    if not weighted_ates:
        return {"ate": float("nan"), "n_matched": 0, "n_bins_used": 0}

    total = sum(n for _, n in weighted_ates)
    ate = float(sum(a * n for a, n in weighted_ates) / total)
    return {
        "ate": ate,
        "n_matched": total,
        "n_bins_used": len(weighted_ates),
        "treated_share": float(treatment.mean()),
    }


def validate_policy(
    logs: pd.DataFrame,
    candidate_propensity: np.ndarray,
    *,
    logging_propensity: np.ndarray | None = None,
    reward_predictions: np.ndarray | None = None,
    n_actions: int | None = None,
) -> dict:
    """
    Compare OPE on exploited impressions vs. unbiased holdout IPS,
    and report propensity-score matching + per-tier errors.
    """
    candidate_propensity = np.asarray(candidate_propensity, dtype=float)
    if logging_propensity is None:
        logging_propensity = logs["logging_propensity"].to_numpy(dtype=float)
    else:
        logging_propensity = np.asarray(logging_propensity, dtype=float)

    holdout_mask = logs["is_randomized"].to_numpy()
    exploit_mask = ~holdout_mask

    holdout_gt = holdout_unbiased_value(
        logs.loc[holdout_mask],
        candidate_propensity[holdout_mask],
        n_actions=n_actions,
    )

    exploit_logs = logs.loc[exploit_mask]
    ope_results: list[OPEResult] = []
    if not exploit_logs.empty:
        exploit_frame = exploit_logs.copy()
        exploit_frame["logging_propensity"] = logging_propensity[exploit_mask]
        mu = reward_predictions[exploit_mask] if reward_predictions is not None else None
        ope_results = run_all_estimators(
            exploit_frame,
            candidate_propensity[exploit_mask],
            reward_predictions=mu,
        )

    compared = compare_ope_to_holdout(ope_results, holdout_gt.value) if ope_results else []

    k = n_actions if n_actions is not None else n_actions_from_logs(logs)
    treatment = candidate_propensity > (1.0 / k)
    psm = propensity_stratified_ate(
        logs["reward"].to_numpy(dtype=float),
        treatment,
        logging_propensity,
    )

    stratified: dict[str, dict] = {}
    for tier in ("head", "mid", "tail"):
        tier_mask = (logs["popularity_tier"] == tier).to_numpy()
        tier_holdout = holdout_mask & tier_mask
        tier_exploit = exploit_mask & tier_mask
        if tier_holdout.sum() == 0 or tier_exploit.sum() == 0:
            continue
        gt = holdout_unbiased_value(
            logs.loc[tier_holdout],
            candidate_propensity[tier_holdout],
            n_actions=n_actions,
        )
        tier_frame = logs.loc[tier_exploit].copy()
        tier_frame["logging_propensity"] = logging_propensity[tier_exploit]
        mu = reward_predictions[tier_exploit] if reward_predictions is not None else None
        tier_ope = run_all_estimators(
            tier_frame,
            candidate_propensity[tier_exploit],
            reward_predictions=mu,
        )
        stratified[tier] = {
            "holdout_ips": gt.to_dict(),
            "estimators": compare_ope_to_holdout(tier_ope, gt.value),
        }

    snips_row = next((r for r in compared if r["estimator"] == "SNIPS"), None)
    ips_values = [r["estimate"] for r in compared]
    gt_values = [r["ground_truth"] for r in compared]

    return {
        "holdout_ground_truth": holdout_gt.to_dict(),
        "n_holdout": int(holdout_mask.sum()),
        "n_exploit": int(exploit_mask.sum()),
        "estimators_vs_holdout": compared,
        "snips_absolute_bias": snips_row["absolute_bias"] if snips_row else None,
        "rmse_across_estimators": rmse(np.array(ips_values), np.array(gt_values))
        if ips_values
        else None,
        "propensity_score_matching": psm,
        "stratified_by_tier": stratified,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate OPE against randomized holdout")
    parser.add_argument("--logs", type=Path, default=Path("data/processed/bandit_logs.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/validation_report.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from src.ope.evaluate import compute_candidate_propensities
    from src.simulator.bandit import PolicyWeights

    logs = pd.read_parquet(args.logs)
    candidate_p = compute_candidate_propensities(logs, PolicyWeights.candidate_v2(), seed=args.seed)
    report = validate_policy(logs, candidate_p)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote report to {args.output}")


if __name__ == "__main__":
    main()
