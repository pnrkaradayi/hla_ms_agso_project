"""Project configuration for the HLA MS-AGSO pipeline."""

from pathlib import Path

# Default sheet-name guesses. The loader is case-insensitive.
SHEET_NAMES = {
    "genomic": "Genomic",
    "nucleotide": "Nucleotide",
    "protein": "Protein",
}

TARGET_COLUMN = "Disease_Id"
DISEASE_LABEL_COLUMN = "Disease_Label"
BINARY_LABEL_COLUMN = "Binary_Label"
ID_COLUMN = "record_ID"

GENOMIC_CATEGORICAL_COLUMNS = [
    "Gene",
    "Allele",
    "Allele_Group",
    "Allele_Subtype",
]

FASTA_COLUMN = "FASTA"

# Feature extraction parameters
DNA_K_VALUES = [3, 4]
PROTEIN_K_VALUES = [1, 2]
MAX_DNA_KMER_FEATURES = 700
MAX_PROTEIN_KMER_FEATURES = 700

# Modelling parameters
TEST_SIZE = 0.20
RANDOM_STATE = 42
CV_FOLDS = 5
SCORING = "f1_macro"

# MS-AGSO parameters. Increase these for final publication runs.
MS_AGSO_PARAMS = {
    "max_initial_features": 450,
    "n_clusters": 8,
    "population_size": 16,
    "ga_generations": 8,
    "pso_particles": 16,
    "pso_iterations": 8,
    "mutation_rate": 0.08,
    "feature_penalty": 0.015,
    "redundancy_threshold": 0.96,
    "random_state": RANDOM_STATE,
}

OUTPUT_DIR = Path("outputs")
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"

for directory in [OUTPUT_DIR, TABLE_DIR, FIGURE_DIR, MODEL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
