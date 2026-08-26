# No Peeking: corrected findings

This copy supersedes the archived validation-selected analysis in `results/`.
The original desktop folder was not modified.

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
- At 4 bits, rotation plus uniform has the highest mean AUC on all four tasks.
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
cd paper && latexmk -pdf dfq_medmnist.tex
```

The corrected result rows include the SHA-256 digest of the checkpoint used for
each evaluation. Legacy CSVs are retained only to preserve the earlier analysis.
