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

Statistical scripts are being adapted for
portable execution from this repository.
