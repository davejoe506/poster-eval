"""FastAPI endpoint tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.pipeline.tasks import persist_policy_evaluations


def _seed_report() -> dict:
    return {
        "candidate_policy": "bandit_v2",
        "logging_policy": {
            "snips_value": 0.12,
            "estimators": [
                {"estimator": "IPS", "value": 0.12, "std_error": 0.02},
                {"estimator": "SNIPS", "value": 0.12, "std_error": 0.015},
                {"estimator": "DR", "value": 0.121, "std_error": 0.014},
            ],
        },
        "candidate_results_oracle_propensity": [
            {"estimator": "IPS", "value": 0.142, "std_error": 0.031},
            {"estimator": "SNIPS", "value": 0.138, "std_error": 0.024},
            {"estimator": "DR", "value": 0.135, "std_error": 0.019},
        ],
        "stratified_by_tier": {
            "head": [
                {"estimator": "SNIPS", "value": 0.16, "std_error": 0.02},
                {"estimator": "IPS", "value": 0.17, "std_error": 0.03},
            ],
            "mid": [{"estimator": "SNIPS", "value": 0.13, "std_error": 0.02}],
            "tail": [{"estimator": "SNIPS", "value": 0.11, "std_error": 0.03}],
        },
        "propensity_model": {"metrics": {"coverage_above_floor": 0.995}},
        "validation_oracle_propensity": {
            "holdout_ground_truth": {"value": 0.133},
            "snips_absolute_bias": 0.005,
            "rmse_across_estimators": 0.008,
            "n_holdout": 200,
            "n_exploit": 1800,
        },
    }


@pytest.fixture
def api_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db = tmp_path / "postereval.db"
    persist_policy_evaluations(_seed_report(), db)
    monkeypatch.setenv("POSTEREVAL_DB_PATH", str(db))
    return db


@pytest.fixture
def client(api_db: Path) -> TestClient:
    from src.api.main import app

    return TestClient(app)


@pytest.fixture
def empty_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("POSTEREVAL_DB_PATH", str(tmp_path / "missing.db"))
    from src.api.main import app

    return TestClient(app)


def test_health_ok(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["db_available"] is True
    assert body["service"] == "postereval-api"


def test_health_reports_missing_db(empty_client: TestClient):
    response = empty_client.get("/health")
    assert response.status_code == 200
    assert response.json()["db_available"] is False


def test_policies_compare_from_sqlite(client: TestClient):
    response = client.get("/policies/compare")
    assert response.status_code == 200
    body = response.json()
    assert body["policy_id"] == "bandit_v2"
    assert set(body["estimators"]) >= {"IPS", "SNIPS", "DR"}
    assert body["estimators"]["SNIPS"]["value"] == pytest.approx(0.138)
    assert body["logging_policy_value"] == pytest.approx(0.12)
    assert body["lift_vs_logging"] == pytest.approx(0.018)
    assert body["recommendation"] == "DEPLOY_WITH_CAUTION"


def test_policies_compare_unknown_policy(client: TestClient):
    response = client.get("/policies/compare", params={"policy_id": "does_not_exist"})
    assert response.status_code == 404


def test_validation_holdout(client: TestClient):
    response = client.get("/validation/holdout")
    assert response.status_code == 200
    body = response.json()
    assert body["candidate_policy"] == "bandit_v2"
    assert body["snips_absolute_bias"] == pytest.approx(0.005)
    assert body["rmse_across_estimators"] == pytest.approx(0.008)
    assert body["coverage_above_floor"] == pytest.approx(0.995)
    assert body["n_holdout"] == 200
    assert body["n_exploit"] == 1800
    assert body["ground_truth_value"] == pytest.approx(0.133)


def test_reports_stratified_by_tier(client: TestClient):
    response = client.get("/reports/stratified")
    assert response.status_code == 200
    body = response.json()
    assert set(body["tiers"]) == {"head", "mid", "tail"}
    assert body["tiers"]["tail"]["SNIPS"]["value"] == pytest.approx(0.11)
    assert body["tiers"]["head"]["SNIPS"]["value"] == pytest.approx(0.16)


def test_compare_404_when_db_missing(empty_client: TestClient):
    response = empty_client.get("/policies/compare")
    assert response.status_code == 404


def test_seeded_sqlite_has_logging_and_candidate_rows(api_db: Path):
    conn = sqlite3.connect(api_db)
    policies = {
        row[0]
        for row in conn.execute("SELECT DISTINCT candidate_policy FROM policy_evaluations")
    }
    conn.close()
    assert policies == {"bandit_v2", "logging_v1"}
