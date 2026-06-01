# HLA MS-AGSO Autoimmune Disease Classification Project

This project implements an end-to-end Python pipeline for HLA-based multiclass classification of autoimmune disease groups:

- Disease_Id = 1: Celiac disease
- Disease_Id = 2: Type 1 Diabetes (T1D)
- Disease_Id = 3: Multiple Sclerosis (MS)

The code supports an Excel file with three sheets:

- `Genomic`
- `Nucleotide`
- `Protein`

Expected columns in each sheet:

`record_ID, Gene, Allele, Allele_Group, Allele_Subtype, FASTA, Disease_Label, Binary_Label, Disease_Id`

## What the pipeline does

1. Loads the three Excel sheets.
2. Uses `Disease_Id` as the multiclass target.
3. Excludes leakage columns: `Disease_Label`, `Binary_Label`, and `record_ID` from model inputs.
4. Builds separate feature layers:
   - Genomic categorical features: `Gene`, `Allele`, `Allele_Group`, `Allele_Subtype`
   - Genomic FASTA-derived DNA k-mer features
   - Nucleotide FASTA-derived DNA k-mer features
   - Protein FASTA-derived amino-acid features
   - Combined multi-layer representation
5. Runs baseline classifiers.
6. Runs the proposed MS-AGSO feature-selection pipeline.
7. Saves result tables and figures.

## Installation

```bash
pip install -r requirements.txt
```

## Run

Place your Excel file in the project folder or pass its path directly.

```bash
python main.py --excel_path "Allels_Combined.xlsx"
```

Outputs will be saved in:

```text
outputs/tables/
outputs/figures/
outputs/models/
```

## Notes

- FASTA columns are not used as raw text. They are converted into numeric k-mer or amino-acid-derived features.
- For manuscript writing, use the generated CSV tables for Results.
- This implementation is written to be clear and modifiable, not just minimal.
