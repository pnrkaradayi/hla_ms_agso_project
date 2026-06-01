"""Data loading and validation for the HLA MS-AGSO project."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from config import (
    BINARY_LABEL_COLUMN,
    DISEASE_LABEL_COLUMN,
    FASTA_COLUMN,
    GENOMIC_CATEGORICAL_COLUMNS,
    ID_COLUMN,
    SHEET_NAMES,
    TARGET_COLUMN,
)
from src.utils import clean_column_names, make_unique_index


REQUIRED_COLUMNS = [
    ID_COLUMN,
    "Gene",
    "Allele",
    "Allele_Group",
    "Allele_Subtype",
    FASTA_COLUMN,
    DISEASE_LABEL_COLUMN,
    BINARY_LABEL_COLUMN,
    TARGET_COLUMN,
]


class HLADataError(ValueError):
    """Raised when the HLA Excel file is missing required structure."""


def _find_sheet_name(excel_path: str | Path, desired_name: str) -> str:
    """Find a sheet name case-insensitively."""
    xls = pd.ExcelFile(excel_path)
    available = xls.sheet_names
    lower_map = {s.lower().strip(): s for s in available}
    key = desired_name.lower().strip()
    if key in lower_map:
        return lower_map[key]

    # Allow fuzzy contains matching, e.g. "genomic_data".
    for sheet in available:
        if key in sheet.lower().strip():
            return sheet

    raise HLADataError(
        f"Sheet '{desired_name}' was not found. Available sheets: {available}"
    )


def _standardize_disease_id(df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    """Convert Disease_Id to integer category labels."""
    out = df.copy()
    if TARGET_COLUMN not in out.columns:
        raise HLADataError(f"'{TARGET_COLUMN}' column is missing in sheet '{sheet_name}'.")

    out[TARGET_COLUMN] = pd.to_numeric(out[TARGET_COLUMN], errors="coerce")
    out = out.dropna(subset=[TARGET_COLUMN]).copy()
    out[TARGET_COLUMN] = out[TARGET_COLUMN].astype(int)

    valid = {1, 2, 3}
    observed = set(out[TARGET_COLUMN].unique())
    invalid = observed - valid
    if invalid:
        raise HLADataError(
            f"Invalid Disease_Id values in sheet '{sheet_name}': {sorted(invalid)}. "
            f"Expected only 1=Celiac, 2=T1D, 3=MS."
        )
    return out


def _validate_columns(df: pd.DataFrame, sheet_name: str) -> None:
    """Warn about missing columns and fail if essential columns are missing."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise HLADataError(
            f"Sheet '{sheet_name}' is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )


def _clean_sheet(df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    """Clean a loaded sheet."""
    df = clean_column_names(df)
    _validate_columns(df, sheet_name)
    df = _standardize_disease_id(df, sheet_name)

    for col in GENOMIC_CATEGORICAL_COLUMNS + [FASTA_COLUMN, DISEASE_LABEL_COLUMN]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    df = make_unique_index(df, preferred_id_col=ID_COLUMN)
    return df


def load_hla_excel(excel_path: str | Path) -> Dict[str, pd.DataFrame]:
    """Load Genomic, Nucleotide and Protein sheets from an Excel file.

    Parameters
    ----------
    excel_path:
        Path to the Excel file containing the HLA sheets.

    Returns
    -------
    dict
        Dictionary with keys: genomic, nucleotide, protein.
    """
    excel_path = Path(excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file was not found: {excel_path}")

    loaded = {}
    for key, desired_sheet in SHEET_NAMES.items():
        actual_sheet = _find_sheet_name(excel_path, desired_sheet)
        df = pd.read_excel(excel_path, sheet_name=actual_sheet, engine="openpyxl")
        loaded[key] = _clean_sheet(df, actual_sheet)

    return loaded


def summarize_sheets(sheets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create a summary table for all sheets and disease classes."""
    rows: List[dict] = []
    disease_map = {1: "Celiac disease", 2: "Type 1 Diabetes", 3: "Multiple Sclerosis"}
    for sheet_name, df in sheets.items():
        counts = df[TARGET_COLUMN].value_counts().sort_index()
        for disease_id, count in counts.items():
            rows.append(
                {
                    "sheet": sheet_name,
                    "Disease_Id": int(disease_id),
                    "Disease": disease_map.get(int(disease_id), "Unknown"),
                    "records": int(count),
                    "percentage": round(float(count / len(df) * 100), 2),
                }
            )
    return pd.DataFrame(rows)


def check_label_consistency(sheets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Check whether Disease_Id is consistent across sheets for matching record_ID.

    This is useful when all three sheets represent the same alleles in genomic,
    nucleotide and protein forms.
    """
    frames = []
    for name, df in sheets.items():
        if ID_COLUMN in df.columns:
            frames.append(df[[ID_COLUMN, TARGET_COLUMN]].rename(columns={TARGET_COLUMN: f"{name}_Disease_Id"}))

    if not frames:
        return pd.DataFrame()

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=ID_COLUMN, how="outer")

    disease_cols = [c for c in merged.columns if c.endswith("Disease_Id")]
    merged["consistent"] = merged[disease_cols].nunique(axis=1, dropna=True) <= 1
    return merged


def get_target_from_genomic(sheets: Dict[str, pd.DataFrame]) -> pd.Series:
    """Use Disease_Id from the genomic sheet as the main target."""
    y = sheets["genomic"][TARGET_COLUMN].copy()
    y.name = TARGET_COLUMN
    return y
