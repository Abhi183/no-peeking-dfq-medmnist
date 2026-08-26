# Corrected quantization protocol

Each convolution or linear tensor is reshaped to `(output_channels, fan_in)` and
quantized by row. The first convolution and final classifier remain INT8. Bias,
batch-normalization, and excluded small tensors remain FP16.

At 4 bits the experiment crosses two independent choices:

| | uniform grid | Gaussian Lloyd-Max grid |
|---|---|---|
| no rotation | `quant_symmetric_full_bbit` | `quant_normal_codebook` |
| rotation | `quant_rotated_uniform_bbit` | `quant_turboquant` |

Both grids use exactly `2**b` reconstruction levels and one FP16 scale per row.
For fan-in below 64, rotation is disabled without changing the grid family.

The main evaluation fixes rotation seed 0 independently of checkpoint training
seed. `make_tensor_seed` derives a stable per-tensor seed from rotation seed and
tensor index. The factorial evaluation varies rotation seeds 0 through 4 while
holding each checkpoint fixed.

`quant_uniform_bbit` remains in the source only to reproduce legacy CSVs. It is
not used by the corrected evaluation because it has `2**b - 1` levels.
