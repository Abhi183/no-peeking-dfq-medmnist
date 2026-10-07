# MedMNIST saved representation verification

Completed on all 12 existing checkpoints. No retraining or calibration data were used. The full evaluation produced 384 indexed prediction files, 228 packed representations, and 336 paired-bootstrap interval rows. Forty tests pass; all artifact hashes and recomputed metrics verify.

## Main findings

| Dataset | Uniform 4b | Rotated uniform 4b | Rotated Gaussian 4b | DFQ adaptation 4b |
|---|---:|---:|---:|---:|
| dermamnist | 0.613478 ± 0.025185 | 0.672692 ± 0.039307 | 0.595310 ± 0.069548 | 0.623719 ± 0.074881 |
| pneumoniamnist | 0.684142 ± 0.085217 | 0.840500 ± 0.053940 | 0.790396 ± 0.138809 | 0.731739 ± 0.079345 |
| bloodmnist | 0.770854 ± 0.018756 | 0.812069 ± 0.091807 | 0.796074 ± 0.028975 | 0.831360 ± 0.040749 |
| pathmnist | 0.820563 ± 0.098044 | 0.889953 ± 0.052168 | 0.753684 ± 0.065452 | 0.916092 ± 0.041372 |

Values are mean AUC ± sample SD across three training seeds. The DFQ adaptation has the largest observed means on BloodMNIST and PathMNIST; rotated uniform leads on DermaMNIST and PneumoniaMNIST. This does not establish a universal winner.

Across all 132 saved/archive pairs, maximum absolute differences are AUC 0.00380780, accuracy 0.00961538 and macro F1 0.00843490. Across the 48 four-bit pairs, maxima are AUC 0.00281613, accuracy 0.00648379 and macro F1 0.00402233. These effects include both FP16 scale rounding and rounding excluded parameters. The full-scale rerun matches the original six-decimal CSV values within 5e-7.

## Paired bootstrap

| Dataset | DFQ minus uniform AUC | 95% image-bootstrap interval |
|---|---:|---:|
| dermamnist | +0.010241 | [-0.008836, +0.029044] |
| pneumoniamnist | +0.047597 | [+0.017903, +0.075954] |
| bloodmnist | +0.060506 | [+0.055424, +0.065877] |
| pathmnist | +0.095529 | [+0.092449, +0.098638] |

2,000 class-stratified paired image resamples, seed 20260910. Identical image resamples are used across methods and checkpoints. These marginal percentile intervals are conditional on the three trained checkpoints and observed class counts; they exclude retraining variability, prevalence shifts and patient clustering. They are not multiplicity-adjusted.

## DFQ scope

The matched baseline folds BN, equalizes 13 supported ReLU paths, and applies analytical bias correction on two directly supported pointwise layers. A folding-only control separates those operations from BN folding. No operation crosses hard-swish, residual addition or SE multiplication. High-bias absorption and activation quantization are omitted. This is a restricted weight-only adaptation of [Nagel et al. (2019)](https://arxiv.org/abs/1906.04721), not canonical W8A8 DFQ. Exact adaptations and reproduction commands are in ../../code/STORAGE_STUDY.md.

## Remaining author input

The original manuscript reports a 12-epoch budget. The training script defaults to 15 epochs and learning rate 0.001. Batch counters relative to the cached pretrained model are consistent with selected epochs 6–12 at batch size 128, but do not establish the total epoch budget or learning rate. Please provide the original command or logs. No additional inference test is required for this study. The new code/results have not been pushed to GitHub.

## Files

- metrics.csv: all 384 evaluations and SHA256 hashes.
- predictions/: test-split index, label, logits, probabilities and provenance for each run.
- packed/: actual packed bytes, including FP16 scales/parameters, original buffers and metadata.
- paired_bootstrap.csv: per-checkpoint and mean-of-three paired intervals.
- *_bootstrap_replicates.npz: all bootstrap metric replicates.
- roundtrip_differences.csv: paired archived-precision versus saved-representation metrics.
- protocol.json, bootstrap_protocol.json, requirements-lock.txt: environment and methods.
- dfq_audit.json, scale_audit.json, training_provenance.json, verification.json, test_results.txt: validation and provenance.
- source_manifest.json: exact source file hashes for the delivered implementation.
