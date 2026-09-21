"""Pure contribution reconstruction for the frozen E04 verifier pipelines."""

from __future__ import annotations

import re

import numpy as np


_BASE_WIDTH = 56
_FEATURE_WIDTH = 116


def _sigmoid(raw_margin: np.ndarray) -> np.ndarray:
    """Return a numerically stable sigmoid without invoking model scoring APIs."""
    result = np.empty_like(raw_margin, dtype=float)
    positive = raw_margin >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-raw_margin[positive]))
    exp_margin = np.exp(raw_margin[~positive])
    result[~positive] = exp_margin / (1.0 + exp_margin)
    return result


def _retained_feature_indices(imputer: object, transformed_width: int) -> np.ndarray:
    """Map SimpleImputer output positions back to original raw feature indices."""
    names = getattr(imputer, "get_feature_names_out")()
    input_names = getattr(imputer, "feature_names_in_", None)
    input_positions = (
        {str(name): index for index, name in enumerate(input_names)}
        if input_names is not None else {}
    )
    positions: list[int] = []
    for name in names:
        text = str(name)
        if text in input_positions:
            positions.append(input_positions[text])
            continue
        match = re.fullmatch(r"x(\d+)", text)
        if match is None:
            raise RuntimeError("verifier imputer output cannot be mapped to raw features")
        positions.append(int(match.group(1)))
    indices = np.asarray(positions, dtype=np.int64)
    if (
        indices.shape != (transformed_width,)
        or np.any(indices < 0)
        or np.any(indices >= _FEATURE_WIDTH)
        or len(set(indices.tolist())) != len(indices)
    ):
        raise RuntimeError("verifier imputer retained-feature mapping is invalid")
    return indices


def _probability_identity(raw_margin: np.ndarray, expected: np.ndarray, label: str) -> None:
    if expected.shape != raw_margin.shape:
        raise RuntimeError(f"{label} probability identity has incompatible shape")
    if not np.all(np.isfinite(expected)) or not np.allclose(_sigmoid(raw_margin), expected, atol=1e-6, rtol=0.0):
        raise RuntimeError(f"{label} probability identity does not match frozen replay score")


def explain_fitted_verifiers(
    features: np.ndarray,
    logistic_pipeline: object,
    lgbm_pipeline: object,
    expected_lr: np.ndarray,
    expected_lgbm: np.ndarray,
) -> dict[str, np.ndarray]:
    """Reconstruct fitted verifier margins from their frozen preprocessing paths.

    This intentionally uses no fitting or ``predict_proba`` call: it explains
    the already-stored replay scores and fails closed if either identity moves.
    """
    raw_features = np.asarray(features)
    if raw_features.ndim != 2 or raw_features.shape[1] != _FEATURE_WIDTH:
        raise ValueError("verifier features must be a two-dimensional 116-column matrix")
    expected_lr_array = np.asarray(expected_lr, dtype=float)
    expected_lgbm_array = np.asarray(expected_lgbm, dtype=float)
    if expected_lr_array.shape != (len(raw_features),) or expected_lgbm_array.shape != (len(raw_features),):
        raise ValueError("expected verifier probabilities must have one value per feature row")

    logistic_steps = getattr(logistic_pipeline, "named_steps")
    lr_imputer = logistic_steps["imputer"]
    lr_scaler = logistic_steps["scaler"]
    lr_model = logistic_steps["model"]
    lr_transformed = lr_scaler.transform(lr_imputer.transform(raw_features))
    coefficient = np.asarray(lr_model.coef_, dtype=float)
    intercept = np.asarray(lr_model.intercept_, dtype=float)
    if coefficient.shape != (1, _FEATURE_WIDTH) or intercept.shape != (1,):
        raise RuntimeError("frozen logistic verifier is not a binary 116D model")
    lr_contributions = lr_transformed * coefficient[0]
    lr_base = lr_contributions[:, :_BASE_WIDTH].sum(axis=1)
    lr_context = lr_contributions[:, _BASE_WIDTH:].sum(axis=1)
    lr_bias = np.full(len(raw_features), intercept[0], dtype=float)
    lr_margin = lr_base + lr_context + lr_bias
    if not np.allclose(lr_margin, lr_model.decision_function(lr_transformed), atol=1e-6, rtol=0.0):
        raise RuntimeError("logistic raw-margin identity does not reconstruct decision function")
    _probability_identity(lr_margin, expected_lr_array, "logistic")

    lgbm_steps = getattr(lgbm_pipeline, "named_steps")
    lgbm_imputer = lgbm_steps["imputer"]
    lgbm_model = lgbm_steps["model"]
    lgbm_transformed = lgbm_imputer.transform(raw_features)
    retained = _retained_feature_indices(lgbm_imputer, lgbm_transformed.shape[1])
    booster = getattr(lgbm_model, "booster_", None)
    if booster is None:
        raise RuntimeError("frozen LightGBM verifier has no native booster")
    native_contributions = np.asarray(booster.predict(lgbm_transformed, pred_contrib=True), dtype=float)
    if native_contributions.shape != (len(raw_features), len(retained) + 1):
        raise RuntimeError("LightGBM pred_contrib output does not match retained verifier features")
    feature_contributions = native_contributions[:, :-1]
    lgbm_base = feature_contributions[:, retained < _BASE_WIDTH].sum(axis=1)
    lgbm_context = feature_contributions[:, retained >= _BASE_WIDTH].sum(axis=1)
    lgbm_bias = native_contributions[:, -1]
    lgbm_margin = lgbm_base + lgbm_context + lgbm_bias
    raw_margin = np.asarray(lgbm_model.predict(lgbm_transformed, raw_score=True), dtype=float)
    if raw_margin.shape != lgbm_margin.shape or not np.allclose(lgbm_margin, raw_margin, atol=1e-5, rtol=0.0):
        raise RuntimeError("LightGBM raw-margin identity does not reconstruct frozen model")
    _probability_identity(lgbm_margin, expected_lgbm_array, "LightGBM")

    return {
        "lr_base": lr_base,
        "lr_context": lr_context,
        "lr_bias": lr_bias,
        "lgbm_base": lgbm_base,
        "lgbm_context": lgbm_context,
        "lgbm_bias": lgbm_bias,
    }
