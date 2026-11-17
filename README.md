# Protein Sequence–Structure Benchmark

Reproducibility materials for:

Evaluating Sequence–Structure Understanding
in Modern Protein Foundation Models

## Benchmarks

1. Structure prediction:
   124 shared CAMEO targets, four models.

2. Inverse folding:
   883 shared CATH 4.2 backbones, four models.

3. Sequence–structure co-design:
   2,000 unique successfully evaluated designs,
   four models and five sequence lengths.

## Data

- data/structure_prediction_targets.csv
- data/inverse_folding_targets.csv
- data/codesign_samples.csv
- data/cameo_target_selection.csv
- data/cath_target_selection.csv

## Statistical results

The tables/ directory contains the statistical
comparisons and length-stratified benchmark
summaries reported in the manuscript.

## Evaluation limitations

Co-design computational designability is defined
as scTM >= 0.90 after ESMFold refolding.

The results are conditional on successful
evaluation and do not establish end-to-end
generation yield, experimental foldability,
structural novelty, or diversity.

Complete generation-attempt histories were not
recoverable for every model.

## Reproducibility status

The target-level and sample-level benchmark
results are provided for independent analysis.

The original model inference pipelines require
their respective model implementations,
checkpoints, and computational dependencies.

Portable statistical reanalysis scripts and their input data are provided in this repository and in the tested statistical release archive.

## Reproducibility materials (November 2026)

The `reproducibility/` directory contains:

- `statistics_release_final.zip`: portable statistical reanalysis,
  including reconstruction of inverse-folding sequence recovery.
- `structure_evaluation_portable.zip`: experimental and predicted
  structures and scripts for recalculating TM-score, C-alpha RMSD
  and lDDT-C-alpha for 496 model-target pairs.
- `provenance_release.zip`: evaluated AlphaFold 3 output identities,
  MultiFlow inference configurations, recovered local modifications,
  and available checkpoint identifiers and SHA-256 hashes.

The statistical and structural evaluation archives were independently
tested after extraction. All 496 structural metric records were
successfully recomputed.

### Statistical reanalysis

The tested environment uses Python 3.9.16, NumPy 1.26.0,
Pandas 1.5.2 and SciPy 1.11.4.

See `README_statistics.md` for detailed instructions.

### Structural metric reanalysis

Additional dependencies: Biopython 1.79 and tmtools 0.0.2.

After extracting the structural evaluation archive, run:

    python scripts/evaluate_structure_prediction.py       --base structure_prediction       --reference structure_prediction/evaluation_results/four_model_paired_complete.csv       --out results/historical       --mode historical

This reproduces the archived historical metric calculations.

### Co-design structural data

An additional archive contains 2,000 generated PDB files,
2,000 independently refolded PDB files, and sequence-containing
manifests.

Its public download location will be added after upload to a
stable release service.

Reported designability is conditional on successful evaluation,
depends on ESMFold-based refolding, and does not establish
experimental foldability or end-to-end generation yield.

### Provenance limitations

For AlphaFold 3, the exact evaluated CIF files are identified
by their archived paths and SHA-256 checksums. The historical
rationale for selecting the earlier 8YHN_A output remains
unverified.

For MultiFlow, the release contains the recovered current local
patch and archived benchmark Hydra configuration. The complete
identity of the patch with the historical execution-time working
tree is not independently verified.
