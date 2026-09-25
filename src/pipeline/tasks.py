"""Airflow-free task callables for the PosterEval pipeline.

Imported inside DAG task bodies so Airflow DAG parsing does not load XGBoost.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def postereval_home() -> Path:
    configured = os.environ.get("POSTEREVAL_HOME")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    configured = os.environ.get("POSTEREVAL_DATA_DIR")
    path = Path(configured) if configured else postereval_home() / "data" / "processed"
    path.mkdir(parents=True, exist_ok=True)
    return path


def db_path() -> Path:
    configured = os.environ.get("POSTEREVAL_DB_PATH")
    return Path(configured) if configured else data_dir() / "postereval.db"


def generate_bandit_logs_task(
    *,
    n_impressions: int = 20_000,
    n_titles: int = 200,
    n_users: int = 500,
    seed: int = 42,
    output_name: str = "bandit_logs.parquet",
) -> str:
    from src.simulator.generate_logs import generate_bandit_logs

    output = data_dir() / output_name
    logs = generate_bandit_logs(
        n_impressions=n_impressions,
        n_titles=n_titles,
        n_users=n_users,
        seed=seed,
    )
    logs.to_parquet(output, index=False)
    return str(output)


def train_propensity_model_task(
    logs_path: str,
    *,
    model_type: str = "xgboost",
    seed: int = 42,
    output_name: str = "propensity_metrics.json",
) -> str:
    from src.propensity.train import train_propensity_model

    logs = pd.read_parquet(logs_path)
    artifact = train_propensity_model(logs, model_type=model_type, seed=seed)
    payload = {
        "model_type": artifact.model_type,
        "feature_names": artifact.feature_names,
        "n_classes": artifact.n_classes,
        "metrics": artifact.metrics,
        "logs_path": logs_path,
    }
    output = data_dir() / output_name
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(output)


def run_holdout_validation_task(
    logs_path: str,
    *,
    seed: int = 42,
    output_name: str = "validation_report.json",
) -> str:
    from src.ope.evaluate import compute_candidate_propensities
    from src.simulator.bandit import PolicyWeights
    from src.validation.holdout import validate_policy

    logs = pd.read_parquet(logs_path)
    candidate_p = compute_candidate_propensities(logs, PolicyWeights.candidate_v2(), seed=seed)
    report = validate_policy(logs, candidate_p)
    output = data_dir() / output_name
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return str(output)


def evaluate_ope_policies_task(
    logs_path: str,
    *,
    seed: int = 42,
    propensity_model_type: str = "xgboost",
    output_name: str = "ope_report.json",
) -> str:
    from src.ope.evaluate import evaluate_policies

    logs = pd.read_parquet(logs_path)
    report = evaluate_policies(logs, seed=seed, propensity_model_type=propensity_model_type)
    output = data_dir() / output_name
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    persist_policy_evaluations(report, db_path())
    return str(output)


def persist_policy_evaluations(report: dict, database: Path) -> None:
    """Write OPE, logging baseline, stratified, and holdout rows into SQLite."""
    database.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(database)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS policy_evaluations (
                eval_id             INTEGER PRIMARY KEY AUTOINCREMENT,
                eval_timestamp      TEXT NOT NULL,
                candidate_policy    TEXT NOT NULL,
                estimator           TEXT NOT NULL,
                estimated_value     REAL NOT NULL,
                std_error           REAL,
                popularity_tier     TEXT,
                ground_truth_value  REAL,
                estimation_error    REAL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS validation_metrics (
                run_id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                eval_timestamp          TEXT NOT NULL,
                candidate_policy        TEXT NOT NULL,
                snips_absolute_bias     REAL,
                rmse_across_estimators  REAL,
                coverage_above_floor    REAL,
                n_holdout               INTEGER,
                n_exploit               INTEGER,
                ground_truth_value      REAL
            )
            """
        )
        timestamp = datetime.now(timezone.utc).isoformat()
        policy = report.get("candidate_policy", "bandit_v2")
        validation = report.get("validation_oracle_propensity") or {}
        gt = (validation.get("holdout_ground_truth") or {}).get("value")

        def _insert(policy_id: str, row: dict, tier: str | None) -> None:
            value = float(row["value"])
            error = (value - gt) if gt is not None else None
            conn.execute(
                """
                INSERT INTO policy_evaluations (
                    eval_timestamp, candidate_policy, estimator, estimated_value,
                    std_error, popularity_tier, ground_truth_value, estimation_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    policy_id,
                    row["estimator"],
                    value,
                    row.get("std_error"),
                    tier,
                    gt,
                    error,
                ),
            )

        for row in report.get("candidate_results_oracle_propensity") or report.get("candidate_results") or []:
            _insert(policy, row, None)

        for row in report.get("logging_policy", {}).get("estimators") or []:
            _insert("logging_v1", row, None)

        for tier, results in (report.get("stratified_by_tier") or {}).items():
            for row in results:
                _insert(policy, row, tier)

        coverage = (report.get("propensity_model") or {}).get("metrics", {}).get("coverage_above_floor")
        conn.execute(
            """
            INSERT INTO validation_metrics (
                eval_timestamp, candidate_policy, snips_absolute_bias,
                rmse_across_estimators, coverage_above_floor, n_holdout,
                n_exploit, ground_truth_value
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                policy,
                validation.get("snips_absolute_bias"),
                validation.get("rmse_across_estimators"),
                coverage,
                validation.get("n_holdout"),
                validation.get("n_exploit"),
                gt,
            ),
        )
        conn.commit()
    finally:
        conn.close()
