"""Utility functions used across the HLA MS-AGSO project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it as Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    """Save a dictionary as a JSON file."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def safe_to_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Save a DataFrame to CSV, creating parent folders if needed."""
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def set_global_seed(seed: int = 42) -> None:
    """Set NumPy random seed for reproducibility."""
    np.random.seed(seed)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip spaces from column names and keep names consistent."""
    cleaned = df.copy()
    cleaned.columns = [str(c).strip() for c in cleaned.columns]
    return cleaned


def remove_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate columns while keeping the first occurrence."""
    return df.loc[:, ~df.columns.duplicated()].copy()


def make_unique_index(df: pd.DataFrame, preferred_id_col: str = "record_ID") -> pd.DataFrame:
    """Use record_ID as index if it exists and is unique; otherwise keep row index."""
    out = df.copy()
    if preferred_id_col in out.columns and out[preferred_id_col].notna().all():
        if out[preferred_id_col].is_unique:
            out.index = out[preferred_id_col].astype(str)
        else:
            out.index = [f"row_{i}" for i in range(len(out))]
    else:
        out.index = [f"row_{i}" for i in range(len(out))]
    return out


def dataframe_memory_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Return a simple memory/shape summary for a DataFrame."""
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "memory_mb": float(df.memory_usage(deep=True).sum() / 1024**2),
    }
