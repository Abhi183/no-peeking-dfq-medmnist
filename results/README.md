# Results provenance

## Corrected results used by the revised paper

- `dfq_corrected_fixed.csv`: fixed, zero-data low-bit configurations across all
  12 checkpoints. Rows include checkpoint SHA-256 values.
- `dfq_rotation_factorial.csv`: 4-bit rotated-Gaussian method for three model
  seeds x five independent rotation seeds per dataset.
- `dfq_corrected_summary.md`: generated tables and variance decomposition.

## Archived results

Files named `dfq_final_*` or `dfq_medmnist*` reproduce the earlier experiment.
They are retained for provenance but are not evidence for the revised paper.
That experiment used a `2**b - 1`-level uniform baseline, selected low-bit
configurations with validation AUC, and coupled rotation seed to training seed.
