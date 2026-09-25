"""PosterEval daily OPE pipeline.

Task bodies import src.pipeline.tasks lazily so the scheduler can parse this file
without loading XGBoost / sklearn.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task
from airflow.operators.python import get_current_context

_POSTEREVAL_HOME = Path(os.environ.get("POSTEREVAL_HOME", Path(__file__).resolve().parents[1]))
if str(_POSTEREVAL_HOME) not in sys.path:
    sys.path.insert(0, str(_POSTEREVAL_HOME))

from src.pipeline.graph import (  # noqa: E402
    DAG_ID,
    DAGRUN_TIMEOUT_MINUTES,
    DEFAULT_N_IMPRESSIONS,
    DEFAULT_SEED,
    RETRIES,
    RETRY_DELAY_MINUTES,
    SCHEDULE,
)


@dag(
    dag_id=DAG_ID,
    schedule=SCHEDULE,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args={
        "owner": "postereval",
        "retries": RETRIES,
        "retry_delay": timedelta(minutes=RETRY_DELAY_MINUTES),
        "depends_on_past": False,
    },
    dagrun_timeout=timedelta(minutes=DAGRUN_TIMEOUT_MINUTES),
    max_active_runs=1,
    tags=["postereval", "ope", "causal"],
    params={
        "n_impressions": int(os.environ.get("POSTEREVAL_N_IMPRESSIONS", DEFAULT_N_IMPRESSIONS)),
        "n_titles": int(os.environ.get("POSTEREVAL_N_TITLES", "200")),
        "n_users": int(os.environ.get("POSTEREVAL_N_USERS", "500")),
        "seed": int(os.environ.get("POSTEREVAL_SEED", DEFAULT_SEED)),
        "propensity_model": os.environ.get("POSTEREVAL_PROPENSITY_MODEL", "xgboost"),
    },
    doc_md="""
    Off-policy evaluation pipeline for artwork personalization.

    1. Simulate logged bandit impressions (epsilon-greedy mixture propensity)
    2. Train P(action | context) for IPS weighting
    3. Validate OPE against the randomized holdout
    4. Estimate candidate policy value (IPS / SNIPS / DR) and persist results
    """,
)
def postereval_pipeline():
    @task(task_id="generate_bandit_logs")
    def generate_bandit_logs() -> str:
        from src.pipeline.tasks import generate_bandit_logs_task

        params = get_current_context()["params"]
        return generate_bandit_logs_task(
            n_impressions=int(params["n_impressions"]),
            n_titles=int(params["n_titles"]),
            n_users=int(params["n_users"]),
            seed=int(params["seed"]),
        )

    @task(task_id="train_propensity_model")
    def train_propensity_model(logs_path: str) -> str:
        from src.pipeline.tasks import train_propensity_model_task

        params = get_current_context()["params"]
        train_propensity_model_task(
            logs_path,
            model_type=str(params["propensity_model"]),
            seed=int(params["seed"]),
        )
        return logs_path

    @task(task_id="run_holdout_validation")
    def run_holdout_validation(logs_path: str) -> str:
        from src.pipeline.tasks import run_holdout_validation_task

        params = get_current_context()["params"]
        run_holdout_validation_task(logs_path, seed=int(params["seed"]))
        return logs_path

    @task(task_id="evaluate_ope_policies")
    def evaluate_ope_policies(logs_path: str) -> str:
        from src.pipeline.tasks import evaluate_ope_policies_task

        params = get_current_context()["params"]
        return evaluate_ope_policies_task(
            logs_path,
            seed=int(params["seed"]),
            propensity_model_type=str(params["propensity_model"]),
        )

    logs = generate_bandit_logs()
    logs = train_propensity_model(logs)
    logs = run_holdout_validation(logs)
    evaluate_ope_policies(logs)


postereval_pipeline()
