# Corrected 4-bit headline

Test AUC, mean ± sample standard deviation over model seeds 42, 123, and 456.
The layer policy and rotation seed are fixed before evaluation. All scalar grids
use 16 reconstruction levels.

| Dataset | Uniform | Rotation + uniform | Gaussian | Rotation + Gaussian |
|---|---:|---:|---:|---:|
| DermaMNIST | 0.614 ± 0.024 | **0.673 ± 0.040** | 0.546 ± 0.068 | 0.595 ± 0.070 |
| PneumoniaMNIST | 0.685 ± 0.085 | **0.840 ± 0.054** | 0.689 ± 0.129 | 0.791 ± 0.139 |
| BloodMNIST | 0.771 ± 0.018 | **0.812 ± 0.092** | 0.631 ± 0.097 | 0.796 ± 0.029 |
| PathMNIST | 0.821 ± 0.098 | **0.890 ± 0.053** | 0.741 ± 0.055 | 0.754 ± 0.065 |

Rotation plus uniform has the highest mean AUC on all four tasks. Absolute
accuracy and macro-F1 remain far below FP32, so this is not a deployment claim.

Source: `dfq_corrected_fixed.csv`.
