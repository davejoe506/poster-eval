"""DAG contract — imported by the Airflow file and by tests (no Airflow dependency)."""

from __future__ import annotations

DAG_ID = "postereval_pipeline"
SCHEDULE = "@daily"
OWNER = "postereval"
RETRIES = 2
RETRY_DELAY_MINUTES = 5
DAGRUN_TIMEOUT_MINUTES = 30

TASK_ORDER = (
    "generate_bandit_logs",
    "train_propensity_model",
    "run_holdout_validation",
    "evaluate_ope_policies",
)

PIPELINE_EDGES = tuple(zip(TASK_ORDER, TASK_ORDER[1:]))

DEFAULT_N_IMPRESSIONS = 20_000
DEFAULT_SEED = 42
