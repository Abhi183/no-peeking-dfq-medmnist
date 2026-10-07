# Saved representation and matched DFQ study

Run from the repository root in an isolated Python environment. Exact package versions for the completed experiment are in the accompanying `storage_study/requirements-lock.txt` (Python 3.12.14, PyTorch 2.14.0, NumPy 2.5.3, macOS arm64 MPS).

```sh
PYTHONPATH=code python -m pytest code -q
python code/run_storage_study.py --data /path/to/medmnist_data --output /path/to/new_study
python code/paired_bootstrap.py /path/to/new_study --replicates 2000
```

The data directory must contain `{dataset}_64.npz` for bloodmnist, dermamnist, pathmnist and pneumoniamnist. Only `test_images` and `test_labels` are read. Original checkpoint files are read from `code/checkpoints`; nothing is trained. Compression functions receive model state only. The output directory must be fresh. Runs use MPS when available and CPU otherwise. No network access is needed once dependencies, checkpoints and datasets exist.

## What is measured

The study writes 228 packed model files and 384 prediction files: 11 archived-precision configurations, 11 saved representations, two FP32 folding/equalization controls, and eight saved DFQ/control configurations for each of 12 checkpoints. All saved models are read from disk and loaded into a fresh model instance before inference. `metrics.csv` contains checkpoint, prediction, test-data and packed-file SHA256 hashes. `roundtrip_differences.csv` is a derived comparison of archive versus saved metrics.

Each prediction NPZ contains original test-split `index`, `label`, float32 `logits`, float32 `probs`, dataset and model seed, checkpoint SHA256 and test-array SHA256. Test order is fixed. No image data are copied into prediction files.

## Storage format

`DFQPACK1` magic, little-endian uint64 JSON-header length, UTF-8 header, then contiguous payload. Quantized coordinate indices are packed least-significant bit first at 2, 3, 4 or 8 bits. Every output row has one little-endian IEEE binary16 scale. Code assignment uses the original full-precision scale; only saved FP16 scales are available during decoding. Excluded parameters, including biases and BN affine parameters, are stored in FP16. BN buffers retain checkpoint precision and integer counters remain integers. The full FP32 control retains all original dtypes.

The Gaussian centroids are embedded as round-trippable float64 values. Orthogonal transforms are regenerated from stored per-tensor seeds using NumPy's normal generator and QR, with diagonal-sign normalization. NumPy and its linear algebra backend matter for cross-platform decoding; this is not a portable production codec guarantee. Regeneration is not included in inference timing. File byte counts include metadata, codebooks, buffers and packing padding. Zero FP16 scales caused by underflow are decoded literally, without a hidden higher-precision fallback. Overflow is rejected. There is no integer inference kernel or activation quantization.

## Matched DFQ adaptation

Based on Nagel et al. (2019), https://arxiv.org/abs/1906.04721. This is an explicitly restricted weight-only adaptation, not a reproduction of canonical W8A8 DFQ. All Conv–BN pairs are folded. Cross-layer equalization runs on 13 safe Conv–ReLU–Conv pairs: four adjacent pairs in residual blocks 2 and 3, plus nine SE fc1/fc2 pairs. No scaling crosses hard-swish, residual additions or SE multiplication. Scaling uses channel max-absolute ranges and is iterated up to 100 sweeps or max absolute log-scale below 1e-5, with per-step scales clipped to [1e-3, 1e3].

Analytical bias correction uses BN-derived rectified-normal means only for `features.2.block.2.0` and `features.3.block.2.0`, which are pointwise convolutions directly following BN/ReLU. Their 1x1, zero-padding inputs avoid a spatial-padding approximation. Bias corrections are computed against weights decoded from the saved representation and written into FP16 bias bytes without requantizing weight codes. BC does not compensate error propagated from earlier layers. Unsupported activation-statistics assumptions are not introduced. High-bias absorption and activation quantization are omitted. BN statistics constitute stored model information, a broader constraint than reading only weight tensors.

The BN-folding-only control uses the same tensor eligibility, row granularity, 2/3/4/8-bit widths, INT8-protected first/last eligible layers below eight bits, and storage precision. Folding removes BN parameters and buffers and adds convolution biases, so exact byte sizes differ; the comparison matches quantization policy, not an exactly equal byte budget. FP32 folding and CLE controls test functional preservation.

## Uncertainty

2,000 class-stratified paired image bootstrap samples, RNG seed 20260910. Identical image weights are used across every method and all three checkpoints. Metrics are calculated per checkpoint; the mean-of-three intervals average those metrics, not probabilities or concatenated checkpoint/example pairs. AUC is binary positive-class or macro one-vs-rest. Percentile 95% intervals are conditional on these trained checkpoints and observed class counts, marginal and not adjusted for multiple comparisons. They do not capture training uncertainty, prevalence shifts or patient clustering. Patient identifiers are not present in the exported data. Separate training-seed SDs must remain visible.

## Historical training provenance

The original manuscript states 12 epochs. The script defaults to 15 epochs and learning rate 0.001; its batch size is 128, AdamW weight decay 0.0001, with cosine scheduling and best-validation-AUC selection. The checkpoints do not store optimizer, scheduler, learning rate or epoch budget. Batch counters relative to the cached pretrained V1 weights are consistent with selected epochs 6–12 at batch size 128.

Session records recovered on 2026-10-07 (`results/training_provenance/`) show both runs were launched with `--epochs 12 --size 64` from the committed script, and the recorded test metrics and timings match the checkpoints. The epoch budget is therefore documented. The learning rate is not named in those records and is taken to be the script default.
