# Corrected experiment plan

The implemented protocol is fixed and data-free at compression time.

- [x] Recover all 12 MobileNetV3-Small checkpoints.
- [x] Add a full-cardinality symmetric uniform quantizer.
- [x] Add rotation + uniform and no-rotation + Gaussian cells.
- [x] Separate rotation seeds from checkpoint training seeds.
- [x] Evaluate fixed 2-, 3-, and 4-bit configurations.
- [x] Run the 3 model-seed x 5 rotation-seed factorial experiment.
- [x] Generate paired summaries and corrected figures.
- [x] Rebuild and inspect the paper.

Primary output files:

- `results/dfq_corrected_fixed.csv`
- `results/dfq_rotation_factorial.csv`
- `results/dfq_corrected_summary.md`
- `paper/dfq_medmnist.pdf`
