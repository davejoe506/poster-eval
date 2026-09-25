"""Train P(artwork variant | context, action features) for IPS weighting.

The simulator already logs oracle propensities. We still fit a model because
production logs often lack them (or they go stale). Oracle values are used only
to score the model — log-loss, calibration, and MAE vs. true π_e(a|x).

The logging policy scores *variants*, so a context-only classifier is close to
uniform. We expand each impression into K arm-rows (context + genre emphasis +
variant index), train a binary scorer, then softmax over arms to get π(a|x).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from src.ope.estimators import MIN_PROPENSITY
from src.simulator.bandit import DEVICES, GENRES, ArtworkVariant, build_catalog

ModelType = Literal["xgboost", "logistic"]

_TIER_MAP = {"head": 0, "mid": 1, "tail": 2}


def _parse_context(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return json.loads(value)
    raise TypeError(f"Unsupported context_features type: {type(value)!r}")


def _context_feature_dict(ctx: dict, hour: float) -> dict[str, float]:
    features: dict[str, float] = {}
    title_genres = ctx.get("title_genres") or []
    for genre in GENRES:
        features[f"affinity_{genre}"] = float(ctx.get(f"affinity_{genre}", 0.0))
        features[f"title_genre_{genre}"] = 1.0 if genre in title_genres else 0.0

    device = str(ctx.get("device_type", "web"))
    for d in DEVICES:
        features[f"device_{d}"] = 1.0 if device == d else 0.0

    features["account_tenure_days"] = float(ctx.get("account_tenure_days", 0))
    features["popularity_score"] = float(ctx.get("popularity_score", 0.0))
    features["popularity_tier"] = float(_TIER_MAP.get(str(ctx.get("popularity_tier", "mid")), 1))
    features["hour_of_day"] = float(hour)
    return features


def _artworks_by_title(logs: pd.DataFrame, seed: int) -> dict[int, list[ArtworkVariant]]:
    rng = np.random.default_rng(seed)
    n_titles = int(max(logs["title_id"].max(), logs["title_id"].nunique()))
    k = int(logs["variant_index"].max()) + 1
    _, artworks = build_catalog(rng, n_titles=n_titles, variants_per_title=k)
    by_title: dict[int, list[ArtworkVariant]] = defaultdict(list)
    for artwork in artworks:
        by_title[artwork.title_id].append(artwork)
    return by_title


def extract_features(
    logs: pd.DataFrame,
    *,
    include_action: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Build a numeric feature matrix from logged context (and optional action)."""
    contexts = [_parse_context(v) for v in logs["context_features"].tolist()]
    timestamps = pd.to_datetime(logs["timestamp"])
    hours = timestamps.dt.hour.to_numpy()
    variant_indices = logs["variant_index"].to_numpy(dtype=float) if include_action else None
    emphasess = logs["genre_emphasis"].to_numpy() if include_action and "genre_emphasis" in logs.columns else None

    rows: list[list[float]] = []
    feature_names: list[str] | None = None

    for i, ctx in enumerate(contexts):
        features = _context_feature_dict(ctx, float(hours[i]))
        if include_action and variant_indices is not None:
            features["variant_index"] = float(variant_indices[i])
            if emphasess is not None:
                for genre in GENRES:
                    features[f"emphasis_{genre}"] = 1.0 if emphasess[i] == genre else 0.0

        if feature_names is None:
            feature_names = list(features.keys())
        rows.append([features[name] for name in feature_names])

    assert feature_names is not None
    return np.asarray(rows, dtype=float), feature_names


def expand_arm_features(
    logs: pd.DataFrame,
    *,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Expand each impression into K (context, arm) rows.

    Returns (X, y_chosen, impression_idx, variant_index, feature_names).
    """
    by_title = _artworks_by_title(logs, seed)
    contexts = [_parse_context(v) for v in logs["context_features"].tolist()]
    hours = pd.to_datetime(logs["timestamp"]).dt.hour.to_numpy()
    title_ids = logs["title_id"].to_numpy()
    logged_variant = logs["variant_index"].to_numpy(dtype=int)

    rows: list[list[float]] = []
    y: list[int] = []
    impression_idx: list[int] = []
    variant_idx: list[int] = []
    feature_names: list[str] | None = None

    for i, ctx in enumerate(contexts):
        arms = by_title.get(int(title_ids[i]), [])
        if not arms:
            continue
        base = _context_feature_dict(ctx, float(hours[i]))
        for arm in arms:
            features = dict(base)
            features["variant_index"] = float(arm.variant_index)
            for genre in GENRES:
                features[f"emphasis_{genre}"] = 1.0 if arm.genre_emphasis == genre else 0.0
            if feature_names is None:
                feature_names = list(features.keys())
            rows.append([features[name] for name in feature_names])
            y.append(1 if arm.variant_index == logged_variant[i] else 0)
            impression_idx.append(i)
            variant_idx.append(arm.variant_index)

    assert feature_names is not None
    return (
        np.asarray(rows, dtype=float),
        np.asarray(y, dtype=int),
        np.asarray(impression_idx, dtype=int),
        np.asarray(variant_idx, dtype=int),
        feature_names,
    )


def _softmax_rows(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)


def _clip_propensity(p: np.ndarray, floor: float = MIN_PROPENSITY) -> np.ndarray:
    return np.clip(p, floor, 1.0)


def _arm_scores_to_proba(
    scores: np.ndarray,
    impression_idx: np.ndarray,
    variant_idx: np.ndarray,
    n_impressions: int,
    n_classes: int,
) -> np.ndarray:
    """Scatter per-arm scores into an (n_impressions, K) matrix, then softmax."""
    grouped = np.full((n_impressions, n_classes), -1e9, dtype=float)
    grouped[impression_idx, variant_idx] = scores
    return _softmax_rows(grouped)


@dataclass
class PropensityModel:
    """Binary arm-scorer + softmax → P(variant_index | context)."""

    model: Any
    feature_names: list[str]
    n_classes: int
    model_type: str
    seed: int = 42
    metrics: dict[str, float] = field(default_factory=dict)

    def predict_proba(self, logs: pd.DataFrame) -> np.ndarray:
        X, _, impression_idx, variant_idx, _ = expand_arm_features(logs, seed=self.seed)
        scores = np.asarray(self.model.predict_proba(X)[:, 1], dtype=float)
        return _arm_scores_to_proba(scores, impression_idx, variant_idx, len(logs), self.n_classes)

    def predict_logged_propensity(self, logs: pd.DataFrame) -> np.ndarray:
        """Estimated π_e(a_i | x_i) for the action that was actually logged."""
        proba = self.predict_proba(logs)
        actions = logs["variant_index"].to_numpy(dtype=int)
        n = len(actions)
        clipped_actions = np.clip(actions, 0, self.n_classes - 1)
        estimated = proba[np.arange(n), clipped_actions]
        estimated = np.where((actions >= 0) & (actions < self.n_classes), estimated, MIN_PROPENSITY)
        return _clip_propensity(estimated)


def evaluate_propensity(
    *,
    y_true: np.ndarray,
    proba: np.ndarray,
    estimated_propensity: np.ndarray,
    oracle_propensity: np.ndarray | None = None,
    floor: float = MIN_PROPENSITY,
) -> dict[str, float]:
    """Log-loss, coverage, and (if available) MAE vs. simulator oracle π_e."""
    n_classes = proba.shape[1]
    labels = np.arange(n_classes)
    row_sums = proba.sum(axis=1, keepdims=True)
    row_sums = np.clip(row_sums, 1e-12, None)
    proba_normed = proba / row_sums
    metrics: dict[str, float] = {
        "log_loss": float(log_loss(y_true, proba_normed, labels=labels)),
        "uniform_log_loss": float(-np.log(1.0 / n_classes)),
        "min_propensity": float(estimated_propensity.min()) if len(estimated_propensity) else 0.0,
        "mean_propensity": float(estimated_propensity.mean()) if len(estimated_propensity) else 0.0,
        "coverage_above_floor": float((estimated_propensity > floor).mean())
        if len(estimated_propensity)
        else 0.0,
        "n_eval": float(len(y_true)),
        "n_classes": float(n_classes),
    }
    if oracle_propensity is not None and len(oracle_propensity) == len(estimated_propensity):
        metrics["mae_vs_oracle"] = float(np.mean(np.abs(estimated_propensity - oracle_propensity)))

    calibration = _multiclass_calibration_mae(y_true, proba_normed)
    if calibration is not None:
        metrics["calibration_mae"] = calibration

    return metrics


def _qcut_series(values: np.ndarray, n_bins: int) -> pd.Series:
    """pd.qcut on a ndarray returns a Categorical; wrap so .cat works."""
    return pd.Series(pd.qcut(values, n_bins, duplicates="drop"))


def _multiclass_calibration_mae(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    n_bins: int = 5,
) -> float | None:
    """Mean |predicted P(k) - empirical frequency of class k| across bins and classes."""
    gaps: list[float] = []
    for k in range(proba.shape[1]):
        p_k = proba[:, k]
        y_k = (y_true == k).astype(float)
        if np.unique(p_k).size < 3:
            continue
        try:
            bins = _qcut_series(p_k, n_bins)
        except ValueError:
            continue
        for category in bins.cat.categories:
            mask = (bins == category).to_numpy()
            if mask.sum() < 5:
                continue
            gaps.append(abs(float(p_k[mask].mean()) - float(y_k[mask].mean())))
    if not gaps:
        return None
    return float(np.mean(gaps))


def _fit_arm_scorer(model_type: ModelType, X: np.ndarray, y: np.ndarray, seed: int) -> Any:
    if model_type == "xgboost":
        model = XGBClassifier(
            n_estimators=80,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=seed,
            verbosity=0,
        )
        model.fit(X, y)
        return model
    if model_type == "logistic":
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000)),
            ]
        )
        model.fit(X, y)
        return model
    raise ValueError(f"Unknown model_type: {model_type}")


def train_propensity_model(
    logs: pd.DataFrame,
    *,
    model_type: ModelType = "xgboost",
    test_size: float = 0.2,
    seed: int = 42,
) -> PropensityModel:
    """Fit an arm-scorer; evaluate softmax log-loss on a held-out slice of impressions."""
    if "variant_index" not in logs.columns:
        raise ValueError("logs must include variant_index (action within title)")

    n_classes = int(logs["variant_index"].max()) + 1
    y_actions = logs["variant_index"].to_numpy(dtype=int)
    class_counts = pd.Series(y_actions).value_counts()
    stratify = y_actions if class_counts.min() >= 2 else None

    idx_train, idx_val = train_test_split(
        np.arange(len(logs)), test_size=test_size, random_state=seed, stratify=stratify
    )
    train_logs = logs.iloc[idx_train]
    val_logs = logs.iloc[idx_val]

    X_train, y_train, _, _, feature_names = expand_arm_features(train_logs, seed=seed)
    model = _fit_arm_scorer(model_type, X_train, y_train, seed)

    artifact = PropensityModel(
        model=model,
        feature_names=feature_names,
        n_classes=n_classes,
        model_type=model_type,
        seed=seed,
    )
    val_proba = artifact.predict_proba(val_logs)
    estimated = artifact.predict_logged_propensity(val_logs)
    oracle = val_logs["logging_propensity"].to_numpy(dtype=float)
    artifact.metrics = evaluate_propensity(
        y_true=val_logs["variant_index"].to_numpy(dtype=int),
        proba=val_proba,
        estimated_propensity=estimated,
        oracle_propensity=oracle,
    )
    return artifact


def fit_reward_model(
    logs: pd.DataFrame,
    *,
    seed: int = 42,
) -> np.ndarray:
    """XGBoost μ(x, a) = P(reward=1 | context, logged action) for Doubly Robust."""
    X, _ = extract_features(logs, include_action=True)
    y = logs["reward"].to_numpy(dtype=float)
    model = XGBRegressor(
        n_estimators=60,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        n_jobs=-1,
        random_state=seed,
        verbosity=0,
    )
    model.fit(X, y)
    return np.clip(model.predict(X), 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a propensity model on bandit logs")
    parser.add_argument("--logs", type=Path, default=Path("data/processed/bandit_logs.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/propensity_metrics.json"))
    parser.add_argument("--model-type", choices=("xgboost", "logistic"), default="xgboost")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logs = pd.read_parquet(args.logs)
    artifact = train_propensity_model(logs, model_type=args.model_type, seed=args.seed)

    payload = {
        "model_type": args.model_type,
        "feature_names": artifact.feature_names,
        "n_classes": artifact.n_classes,
        "metrics": artifact.metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"\nWrote metrics to {args.output}")


if __name__ == "__main__":
    main()
