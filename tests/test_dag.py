"""Tests for Airflow DAG integrity and task dependencies."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.pipeline.graph import DAG_ID, PIPELINE_EDGES, SCHEDULE, TASK_ORDER

ROOT = Path(__file__).resolve().parents[1]
DAG_FILE = ROOT / "dags" / "postereval_pipeline.py"


def _task_ids_declared_in_dag_file() -> list[str]:
    tree = ast.parse(DAG_FILE.read_text(encoding="utf-8"))
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg == "task_id" and isinstance(kw.value, ast.Constant):
                found.append((node.lineno, kw.value.value))
    found.sort()
    return [name for _, name in found]


def test_pipeline_contract_is_linear():
    assert TASK_ORDER == (
        "generate_bandit_logs",
        "train_propensity_model",
        "run_holdout_validation",
        "evaluate_ope_policies",
    )
    assert PIPELINE_EDGES == (
        ("generate_bandit_logs", "train_propensity_model"),
        ("train_propensity_model", "run_holdout_validation"),
        ("run_holdout_validation", "evaluate_ope_policies"),
    )
    assert SCHEDULE == "@daily"
    assert DAG_ID == "postereval_pipeline"


def test_dag_file_declares_contract_task_ids_in_order():
    declared = _task_ids_declared_in_dag_file()
    assert declared == list(TASK_ORDER)


def test_dag_file_avoids_heavy_imports_at_parse_time():
    """Scheduler parses DAGs often — keep XGBoost/sklearn inside task bodies."""
    tree = ast.parse(DAG_FILE.read_text(encoding="utf-8"))
    top_level_modules: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level_modules.add(node.module.split(".")[0])
            top_level_modules.add(node.module)
    assert "pandas" not in top_level_modules
    assert "xgboost" not in top_level_modules
    assert "sklearn" not in top_level_modules
    assert "src.pipeline.tasks" not in top_level_modules


def test_persist_policy_evaluations_writes_sqlite(tmp_path):
    from src.pipeline.tasks import persist_policy_evaluations

    db = tmp_path / "postereval.db"
    report = {
        "candidate_policy": "bandit_v2",
        "candidate_results_oracle_propensity": [
            {"estimator": "SNIPS", "value": 0.14, "std_error": 0.02},
            {"estimator": "IPS", "value": 0.15, "std_error": 0.03},
        ],
        "validation_oracle_propensity": {"holdout_ground_truth": {"value": 0.13}},
    }
    persist_policy_evaluations(report, db)

    import sqlite3

    conn = sqlite3.connect(db)
    rows = conn.execute(
        """
        SELECT estimator, estimated_value, estimation_error
        FROM policy_evaluations
        WHERE candidate_policy = 'bandit_v2' AND popularity_tier IS NULL
        """
    ).fetchall()
    conn.close()
    by_estimator = {name: (value, err) for name, value, err in rows}
    assert by_estimator["IPS"][0] == pytest.approx(0.15)
    assert by_estimator["IPS"][1] == pytest.approx(0.02)
    assert by_estimator["SNIPS"][0] == pytest.approx(0.14)
    assert by_estimator["SNIPS"][1] == pytest.approx(0.01)


def test_dagbag_loads_without_import_errors():
    pytest.importorskip("airflow")
    from airflow.models import DagBag

    dagbag = DagBag(dag_folder=str(DAG_FILE.parent), include_examples=False, safe_mode=False)
    assert dagbag.import_errors == {}, f"DAG import errors: {dagbag.import_errors}"
    assert DAG_ID in dagbag.dags


def test_dag_schedule_retries_and_linear_dependencies():
    pytest.importorskip("airflow")
    from airflow.models import DagBag

    dagbag = DagBag(dag_folder=str(DAG_FILE.parent), include_examples=False, safe_mode=False)
    dag = dagbag.get_dag(DAG_ID)
    assert dag is not None
    assert dag.schedule_interval == "@daily" or str(dag.timetable.summary) == "@daily"
    assert dag.default_args.get("retries") == 2
    assert set(dag.task_ids) == set(TASK_ORDER)

    for upstream_id, downstream_id in PIPELINE_EDGES:
        downstream = dag.task_dict[downstream_id]
        assert upstream_id in downstream.upstream_task_ids
        assert downstream.upstream_task_ids == {upstream_id}
