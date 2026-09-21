"""Frozen verifier contribution reconstruction contracts for E04."""

from __future__ import annotations

import numpy as np
import pytest
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.pipeline.challenger.replay import ReplayTrace
from src.pipeline.event_stack import EventMetrics


def _features() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(17)
    train = rng.normal(size=(40, 116))
    train[:, 9] = np.nan
    train[::7, 3] = np.nan
    return train, train[:6].copy()


def _fitted_pipelines() -> tuple[np.ndarray, Pipeline, Pipeline]:
    train, features = _features()
    labels = np.asarray([0, 1] * 20)
    logistic = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(random_state=3)),
    ]).fit(train, labels)
    lgbm = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", LGBMClassifier(n_estimators=4, random_state=3, verbosity=-1)),
    ]).fit(train, labels)
    return features, logistic, lgbm


def test_replay_trace_freezes_116d_verifier_features_and_model_mapping():
    """A caller cannot mutate the sealed raw 116D audit matrix or model view."""
    features = np.zeros((2, 116))
    trace = ReplayTrace(
        fold=0, truths=(), truth_slices={}, raw_macro=(), raw_micro=(), raw_union=(),
        candidate_rows=(), admitted_indices=(), final_indices=(), predictions=(),
        metrics=EventMetrics(0, 0, 0, 0.0, 0.0, 0.0), match_pairs=(),
        verifier_features=features,
        verifier_models={"logistic": object(), "lgbm": object()},
    )
    features[0, 0] = 9.0

    assert trace.verifier_features is not None
    assert trace.verifier_features.flags.writeable is False
    assert trace.verifier_features[0, 0] == 0.0
    with pytest.raises(ValueError):
        trace.verifier_features[0, 0] = 1.0
    with pytest.raises(TypeError):
        trace.verifier_models["extra"] = object()  # type: ignore[index]


def test_explanation_reconstructs_fitted_lr_and_lgbm_with_dropped_imputer_column(monkeypatch):
    """Dropping original column 9 must not shift LGBM base/Context attribution."""
    from src.pipeline.challenger.verifier_attribution import explain_fitted_verifiers

    features, logistic, lgbm = _fitted_pipelines()
    expected_lr = logistic.predict_proba(features)[:, 1]
    expected_lgbm = lgbm.predict_proba(features)[:, 1]
    monkeypatch.setattr(logistic, "fit", lambda *args, **kwargs: pytest.fail("must not fit"))
    monkeypatch.setattr(lgbm, "fit", lambda *args, **kwargs: pytest.fail("must not fit"))

    result = explain_fitted_verifiers(features, logistic, lgbm, expected_lr, expected_lgbm)

    assert set(result) == {"lr_base", "lr_context", "lr_bias", "lgbm_base", "lgbm_context", "lgbm_bias"}
    assert result["lr_base"].shape == (6,)
    assert np.allclose(result["lr_base"] + result["lr_context"] + result["lr_bias"], logistic.decision_function(features), atol=1e-6)
    transformed = lgbm.named_steps["imputer"].transform(features)
    contrib = lgbm.named_steps["model"].booster_.predict(transformed, pred_contrib=True)
    assert contrib.shape[1] == 116  # 115 retained columns plus native bias
    assert np.allclose(result["lgbm_base"] + result["lgbm_context"] + result["lgbm_bias"], lgbm.named_steps["model"].predict(transformed, raw_score=True), atol=1e-5)


def test_explanation_fails_closed_when_stored_probability_disagrees():
    """A stale replay score must abort an attribution instead of silently explaining it."""
    from src.pipeline.challenger.verifier_attribution import explain_fitted_verifiers

    features, logistic, lgbm = _fitted_pipelines()
    expected_lr = logistic.predict_proba(features)[:, 1]
    expected_lgbm = lgbm.predict_proba(features)[:, 1]
    expected_lgbm[0] += 0.01

    with pytest.raises(RuntimeError, match="probability"):
        explain_fitted_verifiers(features, logistic, lgbm, expected_lr, expected_lgbm)
