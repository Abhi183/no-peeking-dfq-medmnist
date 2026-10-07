# Corrected findings

This file records what changed between the first pass at this experiment and
the analysis in the current paper. The archived validation-selected results in
`results/` are superseded.

## What changed

1. The uniform baseline now uses all `2**b` reconstruction levels. The archived
   implementation used `2**b - 1` levels while the Gaussian codebook used
   `2**b`.
2. The 4-bit comparison is a matched 2x2 ablation: rotation on/off crossed with
   uniform/Gaussian scalar grids.
3. Low-bit settings are fixed before test evaluation. No validation split is
   used to choose the layer policy, number of rotations, or rotation seed.
4. Rotation seeds are independent of checkpoint training seeds.
5. A factorial experiment evaluates five rotations on each of the 12 fixed
   checkpoints.

## Results

- INT8 mean AUC stays within 0.002 of FP32 on every dataset (largest
  single-checkpoint shift 0.006) and reduces mean stored size from 6.06 MB to
  1.56 MB.
- Two- and three-bit methods remain at or near chance when pooled across tasks.
- At 4 bits in the in-memory experiment, rotation plus uniform has the highest
  mean AUC on all four tasks. Training-seed SDs are large enough that this is an
  observed ordering, not an established one.
- Rotation raises uniform-grid mean AUC on every dataset, but only 8 of 12
  paired checkpoints improve.
- Rotation raises the Gaussian-codebook AUC on all 12 paired checkpoints, but
  the Gaussian family is weaker than rotation plus uniform downstream.
- Mean within-checkpoint standard deviation over rotation seeds is 0.025
  (DermaMNIST), 0.075 (PneumoniaMNIST), 0.029 (BloodMNIST), and 0.022
  (PathMNIST).

The full numerical summary is in `results/dfq_corrected_summary.md`.

## Reproduction

```bash
python code/corrected_eval.py
python code/corrected_eval.py --factorial
python code/corrected_report.py
```

The corrected result rows include the SHA-256 digest of the checkpoint used for
each evaluation. Legacy CSVs are retained only to preserve the earlier analysis.

## Saved-representation study (September 2026)

A second experiment, paper section 4.6, addressed two gaps in the first.

1. The first experiment reconstructed weights to FP32 without ever writing a
   file. The storage sizes it reported were nominal bit counts. The new study
   packs every configuration to disk with FP16 scales and FP16 excluded
   parameters, reads it back into a fresh model, and scores that. Across all
   132 archive-versus-saved pairs the largest AUC change is 0.0038, and 0.0028
   at four bits. Measured file sizes replace the nominal estimates.
2. There was no data-free comparator from the literature. A restricted
   weight-only adaptation of DFQ (Nagel et al., 2019) was added, plus a
   BN-folding-only control. DFQ* has the largest four-bit mean AUC on
   BloodMNIST and PathMNIST; rotated uniform keeps the lead on DermaMNIST and
   PneumoniaMNIST. The README table reflects this.

A class-stratified paired image bootstrap (2,000 replicates, seed 20260910)
was added to separate test-set sampling uncertainty from training-seed
variability. The two are reported side by side and are different estimands.

Summary files are in `results/storage_study/`. Packed models, prediction
exports, and bootstrap replicates are attached to the `storage-study-v1`
release. Code: `code/saved_codec.py`, `code/matched_dfq.py`,
`code/run_storage_study.py`, `code/paired_bootstrap.py`, documented in
`code/STORAGE_STUDY.md`.
