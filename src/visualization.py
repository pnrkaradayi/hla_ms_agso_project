"""Visualization utilities for Results figures."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.preprocessing import label_binarize

from src.evaluation import DISEASE_NAMES
from src.utils import ensure_dir


def plot_confusion_matrix(cm: np.ndarray, labels: Iterable[int], title: str, output_path: str | Path) -> None:
    """Save a confusion matrix figure."""
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    display_labels = [DISEASE_NAMES.get(int(label), str(label)) for label in labels]
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(results_df: pd.DataFrame, output_path: str | Path, metric: str = "macro_f1") -> None:
    """Save a bar plot comparing feature sets and models."""
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    if results_df.empty:
        return
    df = results_df.sort_values(metric, ascending=False).head(25).copy()
    labels = df["feature_set"] + " | " + df["model"]

    fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.35)))
    ax.barh(labels, df[metric])
    ax.set_xlabel(metric)
    ax.set_ylabel("Feature set | model")
    ax.set_title(f"Top model comparison by {metric}")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_msagso_history(history_df: pd.DataFrame, output_path: str | Path) -> None:
    """Save MS-AGSO optimization history."""
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    if history_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for stage, group in history_df.groupby("stage"):
        x = range(len(group))
        ax.plot(list(x), group["best_fitness"].values, marker="o", label=stage)
    ax.set_xlabel("Iteration within stage")
    ax.set_ylabel("Best fitness")
    ax.set_title("MS-AGSO optimization history")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_selected_feature_heatmap(
    X: pd.DataFrame,
    y: pd.Series,
    selected_features: List[str],
    output_path: str | Path,
    max_features: int = 40,
) -> None:
    """Save a heatmap-like matrix of selected feature means by disease class."""
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    features = [f for f in selected_features if f in X.columns][:max_features]
    if not features:
        return

    mean_table = X[features].copy()
    mean_table["Disease_Id"] = y.values
    class_means = mean_table.groupby("Disease_Id")[features].mean()
    class_means.index = [DISEASE_NAMES.get(int(i), str(i)) for i in class_means.index]

    fig, ax = plt.subplots(figsize=(max(10, len(features) * 0.25), 4))
    im = ax.imshow(class_means.values, aspect="auto")
    ax.set_xticks(np.arange(len(features)))
    ax.set_xticklabels(features, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(class_means.shape[0]))
    ax.set_yticklabels(class_means.index)
    ax.set_title("Mean values of MS-AGSO-selected features by disease class")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(model: object, X_test: pd.DataFrame, y_test: pd.Series, output_path: str | Path) -> None:
    """Save one-vs-rest ROC curves if the model supports probability output."""
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    classes = np.array(sorted(y_test.unique()))

    if not hasattr(model, "predict_proba"):
        return

    try:
        y_score = model.predict_proba(X_test)
        y_bin = label_binarize(y_test, classes=classes)
        if y_bin.shape[1] < 2:
            return

        fig, ax = plt.subplots(figsize=(6, 5))
        for idx, cls in enumerate(classes):
            RocCurveDisplay.from_predictions(
                y_bin[:, idx],
                y_score[:, idx],
                name=f"{DISEASE_NAMES.get(int(cls), str(cls))} vs rest",
                ax=ax,
            )
        ax.set_title("One-vs-rest ROC curves")
        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        return
