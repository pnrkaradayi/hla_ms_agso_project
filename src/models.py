"""Classifier definitions for the HLA MS-AGSO project."""

from __future__ import annotations

from typing import Dict

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import SVC


def get_classifiers(random_state: int = 42) -> Dict[str, object]:
    """Return the baseline classifiers used in the experiments."""
    return {
        "LogisticRegression": Pipeline(
            steps=[
                ("scaler", MaxAbsScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        multi_class="auto",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "SVM_RBF": Pipeline(
            steps=[
                ("scaler", MaxAbsScaler()),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        probability=True,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=random_state),
        "DeepMLP": Pipeline(
            steps=[
                ("scaler", MaxAbsScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(128, 64),
                        activation="relu",
                        solver="adam",
                        alpha=1e-4,
                        learning_rate_init=1e-3,
                        max_iter=500,
                        early_stopping=True,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def get_fast_feature_selection_classifier(random_state: int = 42) -> object:
    """A faster classifier used inside feature selection fitness evaluations."""
    return Pipeline(
        steps=[
            ("scaler", MaxAbsScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1200,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )
