# Corrected fixed-protocol results

All low-bit configurations are fixed before evaluation. The uniform baseline and Gaussian codebook each use 2^b reconstruction levels. Rotation seed 0 is shared across training seeds and is independent of the checkpoint seed.

## Four-bit primary comparison

| dataset | uniform AUC | rotation + Gaussian AUC | paired delta |
|---|---:|---:|---:|
| DermaMNIST | 0.614 ± 0.024 | 0.595 ± 0.070 | -0.019 ± 0.051 |
| PneumoniaMNIST | 0.685 ± 0.085 | 0.791 ± 0.139 | +0.105 ± 0.111 |
| BloodMNIST | 0.771 ± 0.018 | 0.796 ± 0.029 | +0.025 ± 0.031 |
| PathMNIST | 0.821 ± 0.098 | 0.754 ± 0.065 | -0.068 ± 0.100 |

## Four-bit 2x2 ablation

| dataset | uniform | rot. + uniform | Gaussian | rot. + Gaussian |
|---|---:|---:|---:|---:|
| DermaMNIST | 0.614 ± 0.024 | 0.673 ± 0.040 | 0.546 ± 0.068 | 0.595 ± 0.070 |
| PneumoniaMNIST | 0.685 ± 0.085 | 0.840 ± 0.054 | 0.689 ± 0.129 | 0.791 ± 0.139 |
| BloodMNIST | 0.771 ± 0.018 | 0.812 ± 0.092 | 0.631 ± 0.097 | 0.796 ± 0.029 |
| PathMNIST | 0.821 ± 0.098 | 0.890 ± 0.053 | 0.741 ± 0.055 | 0.754 ± 0.065 |

## Rotation-seed variance at four bits

| dataset | mean AUC | mean within-model SD | between-model SD |
|---|---:|---:|---:|
| DermaMNIST | 0.612 | 0.025 | 0.078 |
| PneumoniaMNIST | 0.811 | 0.075 | 0.076 |
| BloodMNIST | 0.770 | 0.029 | 0.041 |
| PathMNIST | 0.747 | 0.022 | 0.063 |
