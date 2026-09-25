"""SQLite accessors for served OPE artifacts."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from src.api.schemas import EstimatorValue

LOGGING_POLICY = "logging_v1"
DEFAULT_CANDIDATE = "bandit_v2"


def db_path() -> Path:
    configured = os.environ.get("POSTEREVAL_DB_PATH")
    if configured:
        return Path(configured)
    home = Path(os.environ.get("POSTEREVAL_HOME", Path(__file__).resolve().parents[2]))
    return home / "data" / "processed" / "postereval.db"


def connect(database: Path | None = None) -> sqlite3.Connection:
    path = database or db_path()
    if not path.exists():
        raise FileNotFoundError(f"PosterEval DB not found at {path}")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def db_available() -> bool:
    path = db_path()
    if not path.exists():
        return False
    try:
        conn = sqlite3.connect(path)
        conn.execute("SELECT 1")
        conn.close()
        return True
    except sqlite3.Error:
        return False


def recommend(lift: float | None, bias: float | None) -> str:
    if lift is None:
        return "INSUFFICIENT_DATA"
    if lift > 0.02 and (bias is None or bias < 0.01):
        return "DEPLOY"
    if lift > 0:
        return "DEPLOY_WITH_CAUTION"
    return "DO_NOT_DEPLOY"


def _latest_timestamp(conn: sqlite3.Connection, policy_id: str) -> str | None:
    row = conn.execute(
        """
        SELECT MAX(eval_timestamp) AS ts
        FROM policy_evaluations
        WHERE candidate_policy = ?
        """,
        (policy_id,),
    ).fetchone()
    return row["ts"] if row and row["ts"] else None


def _estimator_map(rows: list[sqlite3.Row]) -> dict[str, EstimatorValue]:
    return {
        row["estimator"]: EstimatorValue(value=row["estimated_value"], std_error=row["std_error"])
        for row in rows
    }


def fetch_policy_compare(policy_id: str = DEFAULT_CANDIDATE) -> dict:
    conn = connect()
    try:
        ts = _latest_timestamp(conn, policy_id)
        if ts is None:
            raise LookupError(f"No evaluations found for policy '{policy_id}'")

        candidate_rows = conn.execute(
            """
            SELECT estimator, estimated_value, std_error, ground_truth_value
            FROM policy_evaluations
            WHERE candidate_policy = ? AND eval_timestamp = ? AND popularity_tier IS NULL
            """,
            (policy_id, ts),
        ).fetchall()
        if not candidate_rows:
            raise LookupError(f"No overall OPE rows for policy '{policy_id}'")

        logging_snips = conn.execute(
            """
            SELECT estimated_value
            FROM policy_evaluations
            WHERE candidate_policy = ? AND estimator = 'SNIPS'
              AND eval_timestamp = ? AND popularity_tier IS NULL
            """,
            (LOGGING_POLICY, ts),
        ).fetchone()

        snips = next((r for r in candidate_rows if r["estimator"] == "SNIPS"), None)
        logging_value = float(logging_snips["estimated_value"]) if logging_snips else None
        candidate_snips = float(snips["estimated_value"]) if snips else None
        lift = (
            candidate_snips - logging_value
            if candidate_snips is not None and logging_value is not None
            else None
        )
        gt = candidate_rows[0]["ground_truth_value"]
        bias = abs(candidate_snips - gt) if candidate_snips is not None and gt is not None else None

        return {
            "policy_id": policy_id,
            "eval_timestamp": ts,
            "estimators": _estimator_map(candidate_rows),
            "logging_policy_value": logging_value,
            "lift_vs_logging": lift,
            "recommendation": recommend(lift, bias),
            "ground_truth_value": gt,
        }
    finally:
        conn.close()


def fetch_holdout(policy_id: str = DEFAULT_CANDIDATE) -> dict:
    conn = connect()
    try:
        row = conn.execute(
            """
            SELECT *
            FROM validation_metrics
            WHERE candidate_policy = ?
            ORDER BY eval_timestamp DESC
            LIMIT 1
            """,
            (policy_id,),
        ).fetchone()
        if row is None:
            raise LookupError(f"No holdout validation found for policy '{policy_id}'")

        estimator_rows = conn.execute(
            """
            SELECT estimator, estimated_value, std_error, estimation_error, ground_truth_value
            FROM policy_evaluations
            WHERE candidate_policy = ? AND eval_timestamp = ? AND popularity_tier IS NULL
            """,
            (policy_id, row["eval_timestamp"]),
        ).fetchall()

        return {
            "candidate_policy": policy_id,
            "eval_timestamp": row["eval_timestamp"],
            "snips_absolute_bias": row["snips_absolute_bias"],
            "rmse_across_estimators": row["rmse_across_estimators"],
            "coverage_above_floor": row["coverage_above_floor"],
            "n_holdout": row["n_holdout"],
            "n_exploit": row["n_exploit"],
            "ground_truth_value": row["ground_truth_value"],
            "estimators": [dict(r) for r in estimator_rows],
        }
    finally:
        conn.close()


def fetch_stratified(policy_id: str = DEFAULT_CANDIDATE) -> dict:
    conn = connect()
    try:
        ts = _latest_timestamp(conn, policy_id)
        if ts is None:
            raise LookupError(f"No evaluations found for policy '{policy_id}'")

        rows = conn.execute(
            """
            SELECT popularity_tier, estimator, estimated_value, std_error
            FROM policy_evaluations
            WHERE candidate_policy = ? AND eval_timestamp = ? AND popularity_tier IS NOT NULL
            """,
            (policy_id, ts),
        ).fetchall()
        if not rows:
            raise LookupError(f"No stratified OPE rows for policy '{policy_id}'")

        tiers: dict[str, dict[str, EstimatorValue]] = {}
        for row in rows:
            tier = row["popularity_tier"]
            tiers.setdefault(tier, {})[row["estimator"]] = EstimatorValue(
                value=row["estimated_value"],
                std_error=row["std_error"],
            )
        return {"policy_id": policy_id, "eval_timestamp": ts, "tiers": tiers}
    finally:
        conn.close()
