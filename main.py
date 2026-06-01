"""Main entry point for the HLA MS-AGSO autoimmune classification project."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import FIGURE_DIR, MODEL_DIR, MS_AGSO_PARAMS, RANDOM_STATE, TABLE_DIR
from src.data_loader import check_label_consistency, load_hla_excel, summarize_sheets
from src.evaluation import evaluate_multiple_feature_sets, evaluate_single_model, get_best_result
from src.feature_engineering import build_all_feature_sets, summarize_feature_sets
from src.feature_selection import get_baseline_selected_feature_sets
from src.models import get_classifiers
from src.ms_agso import MSAGSOSelector
from src.utils import safe_to_csv, save_json, set_global_seed
from src.visualization import (
    plot_confusion_matrix,
    plot_model_comparison,
    plot_msagso_history,
    plot_roc_curves,
    plot_selected_feature_heatmap,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="HLA MS-AGSO classification pipeline")
    parser.add_argument(
        "--excel_path",
        type=str,
        required=True,
        help="Path to Allels_Combined.xlsx or equivalent Excel file.",
    )
    parser.add_argument(
        "--final_feature_set",
        type=str,
        default="combined_all",
        choices=[
            "genomic_allelic",
            "genomic_fasta",
            "nucleotide",
            "protein",
            "genomic_plus_nucleotide",
            "genomic_plus_protein",
            "combined_all",
            "combined_all_extended",
        ],
        help="Feature set on which MS-AGSO will be applied.",
    )
    return parser.parse_args()


def save_project_summary(args: argparse.Namespace, sheet_summary: pd.DataFrame, feature_summary: pd.DataFrame) -> None:
    """Save a compact summary JSON file."""
    summary = {
        "excel_path": str(Path(args.excel_path).resolve()),
        "final_feature_set": args.final_feature_set,
        "sheet_summary_rows": int(sheet_summary.shape[0]),
        "feature_sets": feature_summary.to_dict(orient="records"),
    }
    save_json(summary, TABLE_DIR / "project_summary.json")


def run_baseline_experiments(feature_sets: dict) -> pd.DataFrame:
    """Run baseline models for all feature sets and save results."""
    baseline_results, fitted_objects = evaluate_multiple_feature_sets(feature_sets, random_state=RANDOM_STATE)
    safe_to_csv(baseline_results, TABLE_DIR / "baseline_all_feature_sets_results.csv")
    plot_model_comparison(
        baseline_results,
        FIGURE_DIR / "baseline_model_comparison_macro_f1.png",
        metric="macro_f1",
    )
    return baseline_results


def run_conventional_feature_selection_baselines(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Run simple conventional feature-selection baselines on the final feature set."""
    selected_sets = get_baseline_selected_feature_sets(X, y, random_state=RANDOM_STATE)
    classifiers = get_classifiers(random_state=RANDOM_STATE)
    preferred_model_name = "LogisticRegression"
    preferred_model = classifiers[preferred_model_name]

    rows = []
    for method_name, X_selected in selected_sets.items():
        metrics, _, _, _, _ = evaluate_single_model(
            model=preferred_model,
            X=X_selected,
            y=y,
            model_name=preferred_model_name,
            feature_set_name=f"final_{method_name}",
            random_state=RANDOM_STATE,
            save_model=False,
        )
        metrics["feature_selection_method"] = method_name
        metrics["selected_feature_count"] = int(X_selected.shape[1])
        rows.append(metrics)

    df = pd.DataFrame(rows)
    safe_to_csv(df, TABLE_DIR / "conventional_feature_selection_baselines.csv")
    return df


def run_msagso_final(X: pd.DataFrame, y: pd.Series, final_feature_set_name: str) -> pd.DataFrame:
    """Run MS-AGSO and evaluate selected features with all classifiers."""
    selector = MSAGSOSelector(**MS_AGSO_PARAMS)
    result = selector.fit_select(X, y)

    selected_feature_df = pd.DataFrame({"selected_feature": result.selected_features})
    safe_to_csv(selected_feature_df, TABLE_DIR / "ms_agso_selected_features.csv")
    safe_to_csv(result.feature_cluster_table, TABLE_DIR / "ms_agso_feature_clusters.csv")
    safe_to_csv(result.history, TABLE_DIR / "ms_agso_optimization_history.csv")
    plot_msagso_history(result.history, FIGURE_DIR / "ms_agso_optimization_history.png")

    summary = {
        "original_feature_count": int(X.shape[1]),
        "candidate_feature_count_after_initial_filter": int(len(result.candidate_features)),
        "selected_feature_count_after_ms_agso": int(len(result.selected_features)),
        "reduction_rate": result.reduction_rate,
        "best_internal_fitness": result.best_score,
    }
    save_json(summary, TABLE_DIR / "ms_agso_summary.json")

    X_selected = X[result.selected_features].copy()

    # Evaluate final MS-AGSO feature subset using all classifiers.
    classifiers = get_classifiers(random_state=RANDOM_STATE)
    rows = []
    final_objects = {}
    for model_name, model in classifiers.items():
        metrics, report_df, cm, fitted_model, split = evaluate_single_model(
            model=model,
            X=X_selected,
            y=y,
            model_name=model_name,
            feature_set_name=f"MSAGSO_{final_feature_set_name}",
            random_state=RANDOM_STATE,
            save_model=True,
        )
        metrics["feature_selection_method"] = "MS-AGSO"
        metrics["selected_feature_count"] = int(X_selected.shape[1])
        rows.append(metrics)
        final_objects[model_name] = {
            "report": report_df,
            "confusion_matrix": cm,
            "model": fitted_model,
            "split": split,
        }

        safe_to_csv(report_df.reset_index().rename(columns={"index": "class"}), TABLE_DIR / f"classification_report_MSAGSO_{model_name}.csv")
        labels = sorted(y.unique())
        plot_confusion_matrix(
            cm,
            labels=labels,
            title=f"MS-AGSO final confusion matrix - {model_name}",
            output_path=FIGURE_DIR / f"confusion_matrix_MSAGSO_{model_name}.png",
        )
        X_train, X_test, y_train, y_test = split
        plot_roc_curves(
            fitted_model,
            X_test,
            y_test,
            output_path=FIGURE_DIR / f"roc_curves_MSAGSO_{model_name}.png",
        )

    final_df = pd.DataFrame(rows)
    safe_to_csv(final_df, TABLE_DIR / "ms_agso_final_model_results.csv")

    # Heatmap for the best final classifier.
    best = get_best_result(final_df, metric="macro_f1")
    plot_selected_feature_heatmap(
        X=X_selected,
        y=y,
        selected_features=result.selected_features,
        output_path=FIGURE_DIR / "heatmap_ms_agso_selected_features_by_disease.png",
        max_features=40,
    )

    return final_df


def main() -> None:
    """Run the complete project pipeline."""
    args = parse_args()
    set_global_seed(RANDOM_STATE)

    print("[1/7] Loading Excel sheets...")
    sheets = load_hla_excel(args.excel_path)

    print("[2/7] Saving dataset summaries...")
    sheet_summary = summarize_sheets(sheets)
    safe_to_csv(sheet_summary, TABLE_DIR / "dataset_sheet_class_distribution.csv")

    label_consistency = check_label_consistency(sheets)
    if not label_consistency.empty:
        safe_to_csv(label_consistency, TABLE_DIR / "label_consistency_across_sheets.csv")

    print("[3/7] Building genomic, nucleotide, protein and combined feature sets...")
    feature_sets = build_all_feature_sets(sheets)
    feature_summary = summarize_feature_sets(feature_sets)
    safe_to_csv(feature_summary, TABLE_DIR / "feature_set_summary.csv")
    save_project_summary(args, sheet_summary, feature_summary)

    print("[4/7] Running baseline classification experiments...")
    baseline_results = run_baseline_experiments(feature_sets)

    print("[5/7] Preparing final feature set for feature selection...")
    X_final, y_final = feature_sets[args.final_feature_set]
    safe_to_csv(
        pd.DataFrame({"target": y_final.values}, index=y_final.index).reset_index(),
        TABLE_DIR / "final_target_vector.csv",
    )

    print("[6/7] Running conventional feature-selection baselines...")
    conventional_fs_results = run_conventional_feature_selection_baselines(X_final, y_final)

    print("[7/7] Running final MS-AGSO feature selection and evaluation...")
    msagso_results = run_msagso_final(X_final, y_final, args.final_feature_set)

    all_final_comparison = pd.concat(
        [
            conventional_fs_results,
            msagso_results,
        ],
        ignore_index=True,
        sort=False,
    )
    safe_to_csv(all_final_comparison, TABLE_DIR / "final_feature_selection_comparison.csv")

    print("\nPipeline completed successfully.")
    print(f"Tables saved to: {TABLE_DIR.resolve()}")
    print(f"Figures saved to: {FIGURE_DIR.resolve()}")
    print(f"Models saved to: {MODEL_DIR.resolve()}")


if __name__ == "__main__":
    main()
