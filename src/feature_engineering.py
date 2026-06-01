"""Feature engineering for genomic, nucleotide and protein HLA representations."""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from config import (
    DNA_K_VALUES,
    FASTA_COLUMN,
    GENOMIC_CATEGORICAL_COLUMNS,
    MAX_DNA_KMER_FEATURES,
    MAX_PROTEIN_KMER_FEATURES,
    PROTEIN_K_VALUES,
    TARGET_COLUMN,
)
from src.utils import remove_duplicate_columns

DNA_ALPHABET = set("ACGT")
PROTEIN_ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")


def clean_fasta_sequence(value: object, alphabet: str = "dna") -> str:
    """Clean FASTA text and return a plain sequence string.

    Parameters
    ----------
    value:
        Raw FASTA cell value. Can include header lines beginning with '>'.
    alphabet:
        'dna' or 'protein'. Unknown characters are removed.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""

    text = str(value).upper().strip()
    if text in {"NAN", "NONE", "NULL", "NA", "N/A", ""}:
        return ""

    # Remove FASTA headers and whitespace.
    lines = [line.strip() for line in text.splitlines() if not line.strip().startswith(">")]
    seq = "".join(lines)
    seq = re.sub(r"\s+", "", seq)

    allowed = DNA_ALPHABET if alphabet == "dna" else PROTEIN_ALPHABET
    seq = "".join([ch for ch in seq if ch in allowed])
    return seq


def sequence_basic_descriptors(sequences: Iterable[str], alphabet: str, prefix: str) -> pd.DataFrame:
    """Extract simple sequence-length and composition descriptors."""
    rows = []
    letters = sorted(DNA_ALPHABET if alphabet == "dna" else PROTEIN_ALPHABET)

    for seq in sequences:
        length = len(seq)
        row = {f"{prefix}_length": length}
        if length == 0:
            for letter in letters:
                row[f"{prefix}_ratio_{letter}"] = 0.0
            if alphabet == "dna":
                row[f"{prefix}_gc_content"] = 0.0
            else:
                row[f"{prefix}_hydrophobic_ratio"] = 0.0
                row[f"{prefix}_charged_ratio"] = 0.0
                row[f"{prefix}_aromatic_ratio"] = 0.0
            rows.append(row)
            continue

        counts = Counter(seq)
        for letter in letters:
            row[f"{prefix}_ratio_{letter}"] = counts.get(letter, 0) / length

        if alphabet == "dna":
            row[f"{prefix}_gc_content"] = (counts.get("G", 0) + counts.get("C", 0)) / length
        else:
            hydrophobic = set("AILMFWV")
            charged = set("DEKRH")
            aromatic = set("FWY")
            row[f"{prefix}_hydrophobic_ratio"] = sum(counts.get(a, 0) for a in hydrophobic) / length
            row[f"{prefix}_charged_ratio"] = sum(counts.get(a, 0) for a in charged) / length
            row[f"{prefix}_aromatic_ratio"] = sum(counts.get(a, 0) for a in aromatic) / length

        rows.append(row)

    return pd.DataFrame(rows).fillna(0.0)


def _kmers(seq: str, k: int) -> List[str]:
    """Return all k-mers from a sequence."""
    if len(seq) < k:
        return []
    return [seq[i : i + k] for i in range(len(seq) - k + 1)]


def kmer_frequency_features(
    sequences: Iterable[str],
    k_values: List[int],
    prefix: str,
    max_features: int,
) -> pd.DataFrame:
    """Create normalized k-mer frequency features.

    The function first scans all sequences and keeps the globally most frequent
    k-mers to avoid a very large feature matrix.
    """
    sequences = list(sequences)
    global_counter: Counter = Counter()
    per_seq_counters: List[Counter] = []

    for seq in sequences:
        seq_counter: Counter = Counter()
        for k in k_values:
            seq_counter.update(_kmers(seq, k))
        per_seq_counters.append(seq_counter)
        global_counter.update(seq_counter)

    selected_kmers = [k for k, _ in global_counter.most_common(max_features)]
    if not selected_kmers:
        return pd.DataFrame(index=range(len(sequences)))

    rows = []
    for counter in per_seq_counters:
        total = sum(counter.values())
        if total == 0:
            rows.append({f"{prefix}_kmer_{kmer}": 0.0 for kmer in selected_kmers})
        else:
            rows.append({f"{prefix}_kmer_{kmer}": counter.get(kmer, 0) / total for kmer in selected_kmers})

    return pd.DataFrame(rows).fillna(0.0)


def build_categorical_genomic_features(genomic_df: pd.DataFrame) -> pd.DataFrame:
    """Build one-hot encoded genomic allele identity features."""
    available = [c for c in GENOMIC_CATEGORICAL_COLUMNS if c in genomic_df.columns]
    if not available:
        raise ValueError("No genomic categorical columns were found.")

    categorical = genomic_df[available].fillna("Unknown").astype(str)
    X = pd.get_dummies(categorical, prefix=available, dtype=float)
    X.index = genomic_df.index
    return remove_duplicate_columns(X)


def build_sequence_features(
    df: pd.DataFrame,
    alphabet: str,
    prefix: str,
    k_values: List[int] | None = None,
    max_kmer_features: int | None = None,
) -> pd.DataFrame:
    """Build numeric features from a FASTA column."""
    if FASTA_COLUMN not in df.columns:
        raise ValueError(f"Column '{FASTA_COLUMN}' was not found.")

    if alphabet == "dna":
        k_values = k_values or DNA_K_VALUES
        max_kmer_features = max_kmer_features or MAX_DNA_KMER_FEATURES
    elif alphabet == "protein":
        k_values = k_values or PROTEIN_K_VALUES
        max_kmer_features = max_kmer_features or MAX_PROTEIN_KMER_FEATURES
    else:
        raise ValueError("alphabet must be 'dna' or 'protein'.")

    sequences = [clean_fasta_sequence(v, alphabet=alphabet) for v in df[FASTA_COLUMN].tolist()]
    basic = sequence_basic_descriptors(sequences, alphabet=alphabet, prefix=prefix)
    kmer = kmer_frequency_features(sequences, k_values=k_values, prefix=prefix, max_features=max_kmer_features)

    X = pd.concat([basic, kmer], axis=1)
    X.index = df.index
    return remove_duplicate_columns(X.fillna(0.0))


def align_feature_matrix_with_target(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """Align X and y by index and return numeric features."""
    common_index = X.index.intersection(y.index)
    X_aligned = X.loc[common_index].copy()
    y_aligned = y.loc[common_index].copy()
    X_aligned = X_aligned.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X_aligned, y_aligned


def build_all_feature_sets(sheets: Dict[str, pd.DataFrame]) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """Build all feature sets required for the Results section.

    Returns a dictionary where each value is (X, y).
    """
    genomic_df = sheets["genomic"]
    nucleotide_df = sheets["nucleotide"]
    protein_df = sheets["protein"]

    # The main target is taken from the genomic sheet.
    y = genomic_df[TARGET_COLUMN].copy()
    y.name = TARGET_COLUMN

    X_genomic_cat = build_categorical_genomic_features(genomic_df)
    X_genomic_seq = build_sequence_features(genomic_df, alphabet="dna", prefix="genomic_fasta")
    X_nucleotide = build_sequence_features(nucleotide_df, alphabet="dna", prefix="nucleotide")
    X_protein = build_sequence_features(protein_df, alphabet="protein", prefix="protein")

    # Align single-layer sets with their own Disease_Id when sheet index differs.
    y_genomic = genomic_df[TARGET_COLUMN]
    y_nucleotide = nucleotide_df[TARGET_COLUMN]
    y_protein = protein_df[TARGET_COLUMN]

    feature_sets: Dict[str, Tuple[pd.DataFrame, pd.Series]] = {}
    feature_sets["genomic_allelic"] = align_feature_matrix_with_target(X_genomic_cat, y_genomic)
    feature_sets["genomic_fasta"] = align_feature_matrix_with_target(X_genomic_seq, y_genomic)
    feature_sets["nucleotide"] = align_feature_matrix_with_target(X_nucleotide, y_nucleotide)
    feature_sets["protein"] = align_feature_matrix_with_target(X_protein, y_protein)

    # Combined models: inner join based on record_ID index if possible; otherwise row index.
    X_gn = X_genomic_cat.join(X_nucleotide, how="inner")
    feature_sets["genomic_plus_nucleotide"] = align_feature_matrix_with_target(X_gn, y)

    X_gp = X_genomic_cat.join(X_protein, how="inner")
    feature_sets["genomic_plus_protein"] = align_feature_matrix_with_target(X_gp, y)

    X_all = X_genomic_cat.join(X_nucleotide, how="inner").join(X_protein, how="inner")
    feature_sets["combined_all"] = align_feature_matrix_with_target(X_all, y)

    # Combined including genomic FASTA too, useful as an extended final experiment.
    X_all_extended = (
        X_genomic_cat.join(X_genomic_seq, how="inner")
        .join(X_nucleotide, how="inner")
        .join(X_protein, how="inner")
    )
    feature_sets["combined_all_extended"] = align_feature_matrix_with_target(X_all_extended, y)

    return feature_sets


def summarize_feature_sets(feature_sets: Dict[str, Tuple[pd.DataFrame, pd.Series]]) -> pd.DataFrame:
    """Summarize feature counts and class counts for each feature set."""
    rows = []
    for name, (X, y) in feature_sets.items():
        rows.append(
            {
                "feature_set": name,
                "n_samples": int(X.shape[0]),
                "n_features": int(X.shape[1]),
                "class_1_celiac": int((y == 1).sum()),
                "class_2_t1d": int((y == 2).sum()),
                "class_3_ms": int((y == 3).sum()),
            }
        )
    return pd.DataFrame(rows)
