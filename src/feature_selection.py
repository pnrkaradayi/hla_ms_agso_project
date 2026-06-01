"""Conventional feature selection baselines."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, VarianceThreshold, mutual_info_classif


def fisher_scores(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Compute Fisher scores for multiclass classification."""
    X_values = X.to_numpy(dtype=float)
    y_values = y.to_numpy()
    classes = np.unique(y_values)
    global_mean = X_values.mean(axis=0)

    numerator = np.zeros(X_values.shape[1], dtype=float)
    denominator = np.zeros(X_values.shape[1], dtype=float)

    for cls in classes:
        Xc = X_values[y_values == cls]
        if Xc.shape[0] == 0:
            continue
        n_c = Xc.shape[0]
        mean_c = Xc.mean(axis=0)
        var_c = Xc.var(axis=0) + 1e-12
        numerator += n_c * (mean_c - global_mean) ** 2
        denominator += n_c * var_c

    scores = numerator / (denominator + 1e-12)
    return pd.Series(scores, index=X.columns).sort_values(ascending=False)


def variance_threshold_selection(X: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """Remove features with variance <= threshold."""
    selector = VarianceThreshold(threshold=threshold)
    arr = selector.fit_transform(X)
    selected_cols = X.columns[selector.get_support()].tolist()
    return pd.DataFrame(arr, columns=selected_cols, index=X.index)


def mutual_information_selection(
    X: pd.DataFrame,
    y: pd.Series,
    k: int | None = None,
    percentile: float = 0.30,
    random_state: int = 42,
) -> pd.DataFrame:
    """Select top features according to mutual information."""
    if k is None:
        k = max(1, int(X.shape[1] * percentile))
    k = min(k, X.shape[1])
    selector = SelectKBest(
        score_func=lambda X_arr, y_arr: mutual_info_classif(
            X_arr, y_arr, discrete_features="auto", random_state=random_state
        ),
        k=k,
    )
    arr = selector.fit_transform(X, y)
    selected_cols = X.columns[selector.get_support()].tolist()
    return pd.DataFrame(arr, columns=selected_cols, index=X.index)


def fisher_score_selection(X: pd.DataFrame, y: pd.Series, k: int | None = None, percentile: float = 0.30) -> pd.DataFrame:
    """Select top features according to Fisher score."""
    if k is None:
        k = max(1, int(X.shape[1] * percentile))
    k = min(k, X.shape[1])
    scores = fisher_scores(X, y)
    selected_cols = scores.head(k).index.tolist()
    return X[selected_cols].copy()


def get_baseline_selected_feature_sets(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Return conventional feature-selection baseline matrices."""
    selected: Dict[str, pd.DataFrame] = {}
    selected["NoFeatureSelection"] = X.copy()
    selected["VarianceThreshold"] = variance_threshold_selection(X, threshold=0.0)
    selected["MutualInformationTop30"] = mutual_information_selection(X, y, percentile=0.30, random_state=random_state)
    selected["FisherScoreTop30"] = fisher_score_selection(X, y, percentile=0.30)
    return selected
