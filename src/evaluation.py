"""Model evaluation utilities for the HLA MS-AGSO project."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import label_binarize

from config import CV_FOLDS, MODEL_DIR, RANDOM_STATE, SCORING, TEST_SIZE
from src.models import get_classifiers
from src.utils import ensure_dir


DISEASE_NAMES = {1: "Celiac", 2: "T1D", 3: "MS"}


def _get_score_matrix(model: object, X: pd.DataFrame) -> np.ndarray | None:
    """Return class probability or decision score matrix if available."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T
        return scores
    return None


def compute_multiclass_auc(y_true: pd.Series, score_matrix: np.ndarray | None, classes: np.ndarray) -> float:
    """Compute macro one-vs-rest ROC-AUC if possible."""
    if score_matrix is None:
        return np.nan
    try:
        y_bin = label_binarize(y_true, classes=classes)
        if y_bin.shape[1] == 1:
            return np.nan
        return float(roc_auc_score(y_bin, score_matrix, average="macro", multi_class="ovr"))
    except Exception:
        return np.nan


def evaluate_single_model(
    model: object,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    feature_set_name: str,
    random_state: int = RANDOM_STATE,
    save_model: bool = False,
) -> Tuple[Dict[str, float], pd.DataFrame, np.ndarray, object, Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
    """Train/test evaluate one model and return metrics/report/confusion matrix."""
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = y.astype(int)

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=random_state,
        stratify=stratify,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    classes = np.array(sorted(y.unique()))
    score_matrix = _get_score_matrix(model, X_test)

    metrics = {
        "feature_set": feature_set_name,
        "model": model_name,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_precision": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "macro_roc_auc": compute_multiclass_auc(y_test, score_matrix, classes),
    }

    report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True, zero_division=0)).T
    cm = confusion_matrix(y_test, y_pred, labels=classes)

    if save_model:
        ensure_dir(MODEL_DIR)
        model_path = Path(MODEL_DIR) / f"{feature_set_name}__{model_name}.joblib"
        joblib.dump(model, model_path)
        metrics["model_path"] = str(model_path)

    return metrics, report_df, cm, model, (X_train, X_test, y_train, y_test)


def cross_validation_score(
    model: object,
    X: pd.DataFrame,
    y: pd.Series,
    scoring: str = SCORING,
    cv_folds: int = CV_FOLDS,
) -> float:
    """Compute stratified cross-validation score."""
    n_splits = min(cv_folds, y.value_counts().min())
    if n_splits < 2:
        return np.nan
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=None)
    return float(np.nanmean(scores))


def evaluate_feature_set(
    X: pd.DataFrame,
    y: pd.Series,
    feature_set_name: str,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Evaluate all baseline classifiers on one feature set."""
    results = []
    fitted_objects: Dict[str, object] = {}
    classifiers = get_classifiers(random_state=random_state)

    for model_name, model in classifiers.items():
        metrics, report_df, cm, fitted_model, split = evaluate_single_model(
            model=model,
            X=X,
            y=y,
            model_name=model_name,
            feature_set_name=feature_set_name,
            random_state=random_state,
            save_model=False,
        )
        metrics["cv_macro_f1"] = cross_validation_score(fitted_model, X, y, scoring="f1_macro")
        results.append(metrics)
        fitted_objects[model_name] = {
            "model": fitted_model,
            "report": report_df,
            "confusion_matrix": cm,
            "split": split,
        }

    return pd.DataFrame(results), fitted_objects


def evaluate_multiple_feature_sets(
    feature_sets: Dict[str, Tuple[pd.DataFrame, pd.Series]],
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, object]]]:
    """Evaluate all models for all feature sets."""
    all_results = []
    all_objects: Dict[str, Dict[str, object]] = {}

    for feature_set_name, (X, y) in feature_sets.items():
        if X.shape[0] < 6 or y.nunique() < 2:
            continue
        result_df, fitted_objects = evaluate_feature_set(X, y, feature_set_name, random_state=random_state)
        all_results.append(result_df)
        all_objects[feature_set_name] = fitted_objects

    if not all_results:
        return pd.DataFrame(), all_objects
    return pd.concat(all_results, ignore_index=True), all_objects


def get_best_result(results_df: pd.DataFrame, metric: str = "macro_f1") -> pd.Series:
    """Return the best row according to a metric."""
    if results_df.empty:
        raise ValueError("No results were produced.")
    return results_df.sort_values(metric, ascending=False).iloc[0]
