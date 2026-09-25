"""Tests for propensity model training."""

import numpy as np

from src.ope.estimators import MIN_PROPENSITY
from src.ope.evaluate import evaluate_policies
from src.propensity.train import extract_features, train_propensity_model
from src.simulator.generate_logs import generate_bandit_logs


def test_extract_features_includes_spec_columns():
    logs = generate_bandit_logs(n_impressions=80, n_titles=12, n_users=20, seed=1)
    X, names = extract_features(logs)
    assert X.shape[0] == 80
    assert "hour_of_day" in names
    assert "popularity_tier" in names
    assert any(n.startswith("device_") for n in names)
    assert any(n.startswith("affinity_") for n in names)
    assert any(n.startswith("title_genre_") for n in names)


def test_xgboost_propensity_beats_uniform_and_is_clipped():
    logs = generate_bandit_logs(n_impressions=2500, n_titles=40, n_users=80, seed=7)
    model = train_propensity_model(logs, model_type="xgboost", seed=7)

    assert model.metrics["log_loss"] < model.metrics["uniform_log_loss"]
    assert model.metrics["coverage_above_floor"] >= 0.99

    estimated = model.predict_logged_propensity(logs)
    assert np.isfinite(estimated).all()
    assert (estimated >= MIN_PROPENSITY).all()
    assert (estimated <= 1.0).all()


def test_logistic_propensity_fallback():
    logs = generate_bandit_logs(n_impressions=1200, n_titles=20, n_users=40, seed=3)
    model = train_propensity_model(logs, model_type="logistic", seed=3)
    estimated = model.predict_logged_propensity(logs)
    assert estimated.shape == (len(logs),)
    assert 0.0 < float(estimated.mean()) < 1.0


def test_evaluate_policies_includes_propensity_and_validation():
    logs = generate_bandit_logs(n_impressions=800, n_titles=20, n_users=40, seed=4)
    report = evaluate_policies(logs, seed=4, propensity_model_type="logistic")
    assert "propensity_model" in report
    assert "validation_oracle_propensity" in report
    assert "candidate_results_estimated_propensity" in report
    assert report["validation_oracle_propensity"]["snips_absolute_bias"] is not None
    assert set(report["stratified_by_tier"]).issubset({"head", "mid", "tail"})
