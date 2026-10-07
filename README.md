# An empirical study of data-free weight quantization on MedMNIST

Code, checkpoints, and result files for the paper
[`paper/dfq_medmnist_empirical_study.pdf`](paper/dfq_medmnist_empirical_study.pdf)
by Abhishek Shekhar and Abhey Singh Guram.

This is a measurement study, not a new quantization method. We take twelve
MobileNetV3-Small checkpoints (four MedMNIST tasks, three training seeds each),
quantize their weights without any calibration inputs, and measure what happens
to test AUC, accuracy, and macro F1. Every number in the paper can be traced to
a checkpoint hash and a result row in this repository.

## What we found

INT8 weight rounding is close to harmless on these checkpoints. Dataset-level
mean AUC moves by at most 0.0018 relative to FP32, and that holds after the
model is actually written to disk at the stated precision and read back.

Two- and three-bit weights collapse to near-chance AUC for both scalar families
we tested.

Four-bit results depend on the dataset and the method, and no single method
wins everywhere. On the saved representation (Table 7 in the paper):

| Dataset | Uniform | Rotated uniform | Gaussian | Rotated Gaussian | DFQ* |
|---|---:|---:|---:|---:|---:|
| DermaMNIST | 0.613 | **0.673** | 0.546 | 0.595 | 0.624 |
| PneumoniaMNIST | 0.684 | **0.840** | 0.688 | 0.790 | 0.732 |
| BloodMNIST | 0.771 | 0.812 | 0.632 | 0.796 | **0.831** |
| PathMNIST | 0.821 | 0.890 | 0.741 | 0.754 | **0.916** |

Test AUC, mean over three training seeds, four bits per weight. DFQ* is a
restricted weight-only adaptation of Nagel et al. (2019): BN folding,
cross-layer equalization on 13 safe paths, and analytical bias correction on
two pointwise layers. It is not canonical W8A8 DFQ.

Training-seed standard deviations at four bits range from 0.02 to 0.14 AUC, so
the observed ordering above is not a statistical ranking. Rotation seed alone
swings a single checkpoint by up to 0.32 AUC on PneumoniaMNIST. Choosing a
rotation by weight reconstruction error improves AUC on five checkpoints and
worsens it on six. Low reconstruction error does not predict downstream
performance here.

AUC also hides large losses in classification. Rotated-uniform four-bit mean
accuracy is 0.50 to 0.76 across datasets against 0.77 to 0.97 for FP32.

None of this is a deployment result. Weights are reconstructed to FP32 and
activations stay in floating point. There are no integer kernels, latency, or
energy measurements, and MedMNIST does not establish clinical performance.

## Repository layout

```
paper/                      the current manuscript
code/                       training, quantizers, evaluation, codec, matched DFQ, tests
code/checkpoints/           the 12 FP32 checkpoints (SHA-256 in results/dfq_corrected_manifest.json)
results/                    result CSVs, manifests, and generated summaries
results/storage_study/      saved-representation study: metrics, bootstrap intervals, audits
FINDINGS.md                 what changed between the first pass and the current analysis
```

Two experiments are reported.

The first (paper sections 4.1 to 4.5) quantizes in memory and reconstructs to
FP32 without rounding scales to FP16. Its rows are in
`results/dfq_corrected_fixed.csv` and `results/dfq_rotation_factorial.csv`.

The second (paper section 4.6) writes every configuration to disk in a packed
format with FP16 scales, reads it back into a fresh model, and scores it. It
also adds the DFQ* comparator and a paired image bootstrap. Its summary files
are in `results/storage_study/`. The 228 packed model files, 384 per-example
prediction files, and bootstrap replicates total 350 MB and are attached to the
GitHub release `storage-study-v1` rather than committed. SHA-256 hashes for
every one of them are in `results/storage_study/metrics.csv`, so a downloaded
archive can be checked against the paper.

## Reproducing the numbers

Dependencies for the original experiment are recorded in
`results/dfq_corrected_manifest.json`; for the storage study, in
`results/storage_study/requirements-lock.txt`. MedMNIST downloads on first use
and is not redistributed here.

```bash
# unit tests (40)
PYTHONPATH=code python -m pytest code -q

# in-memory experiment, sections 4.1 to 4.5
python code/corrected_eval.py
python code/corrected_eval.py --factorial
python code/corrected_report.py

# saved-representation study, section 4.6 (needs a fresh output directory)
python code/run_storage_study.py --data /path/to/medmnist_data --output /path/to/new_study
python code/paired_bootstrap.py /path/to/new_study --replicates 2000
```

`code/STORAGE_STUDY.md` documents the packed file format, the DFQ* adaptation,
and the bootstrap protocol in detail.

## Known gaps

The checkpoints do not record the epoch budget or learning rate used to train
them. The manuscript reports 12 epochs; the script defaults to 15. Batch-norm
counters are consistent with selected epochs 6 to 12 at batch size 128 but do
not settle the question. Treat the script defaults as unconfirmed.

Rotations are regenerated from stored seeds with NumPy's QR, so decoding depends
on the NumPy and LAPACK build. The codec is a research artifact, not a portable
format.

## Archived results

Files named `dfq_final_*` and `dfq_medmnist*` in `results/` come from an earlier
pass that used a `2**b - 1` level uniform baseline, selected low-bit settings on
validation AUC, and tied the rotation seed to the training seed. They are kept
for provenance and are not evidence for the paper. See `FINDINGS.md`.
