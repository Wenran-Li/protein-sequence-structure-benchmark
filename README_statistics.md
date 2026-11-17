# Portable statistical reanalysis (release v2)

This package reanalyzes the archived observations **without running the protein models**.
The definitive statistical workflow was validated using
Python 3.9.16, NumPy 1.26.0, pandas 1.5.2, and SciPy 1.11.4.

The released workflow successfully regenerated the
target-level inverse-folding recovery matrix and
reproduced Tables 3, A1, and A2 consistently with
the revised manuscript.

The statistical analyses use 10,000 target-level
percentile-bootstrap resamples with independent
random-number initialization for each summary or
contrast.

## Environment

Validated existing environment: Python 3.9.16, NumPy 1.26.0, pandas 1.5.2, SciPy 1.11.4. No GPU required. For a new environment, use a compatible Python 3.9 interpreter and install `requirements.txt`. Do not use the system Python 3.14 with these pins.

## Required input data

From release root:

- `data/structure_prediction_targets.csv`: 496 rows; `target_id,model,rmsd,tm_score,lddt_ca`
- `data/cath_mpnn_mapping.csv`: 883 rows; `target_id,length` and provenance fields
- `data/inverse_folding_sequences/esm_if1_cath_sequences.csv`: 4415 rows
- `data/inverse_folding_sequences/gvp_cath_sequences.csv`: 4415 rows
- `data/inverse_folding_sequences/proteinmpnn_cath_sequences.csv`: 4415 rows
- `data/inverse_folding_sequences/pifold_cath_sequences.csv`: 883 rows
- `data/codesign_samples.csv`: 2000 rows

Do not replace missing CSVs with hard-coded or simulated data. The `pdb_path`, `generated_pdb` and `esmfold_pdb` columns document historical provenance and are **not** needed to reproduce the summary statistical tables. They are not portable structure-file locations.

## Commands

Run from the extracted release root (the directory containing `data/` and `scripts/`):

```bash
python scripts/build_inverse_recovery.py --data data --out data/inverse_folding_target_recovery.csv
python scripts/reanalyze.py \
  --structure data/structure_prediction_targets.csv \
  --inverse data/inverse_folding_target_recovery.csv \
  --codesign data/codesign_samples.csv \
  --out results/reanalysis
```

`build_inverse_recovery.py` validates each sequence's residue-by-residue recovery, tests that every target has the correct number of samples, and averages across samples per target. The resulting 883-target matrix is then used for Table 3, Table A2 and Figure 3 statistics. `reanalyze.py` produces `structure_summary.csv`, `table_A1_recomputed.csv`, `table_3_recomputed.csv`, `table_A2_recomputed.csv`, `inverse_length_spearman.csv`, `table_A3_recomputed.csv`, `environment.json`.

## Statistical conventions

Numerically near-zero differences (abs(d) <= 1e-10
in reported metric units) are treated as ties for
the better/worse/tie counts.

The definitive released implementation performs
Wilcoxon signed-rank tests on the original paired
differences.

Mean differences and bootstrap confidence intervals
also use the original unrounded values.

## Verification and consistency

Compare **all** regenerated values with the manuscript Tables 2, 3, A1, A2, A3 and the Figure 3 coefficients. The previously reported tie counts for Table A2 are 4,4,7,1,4,11 in pair order ESM-IF1/PiFold, ESM-IF1/GVP-GNN, ESM-IF1/ProteinMPNN, PiFold/GVP-GNN, PiFold/ProteinMPNN, GVP-GNN/ProteinMPNN. Any remaining p-value or CI differences must be investigated and disclosed rather than overwritten.

## Scope

The release is for reanalysis of observed benchmark metrics. Recomputing coordinates, ESMFold refolding, or model predictions from scratch requires separate model/checkpoint and structural inputs.
