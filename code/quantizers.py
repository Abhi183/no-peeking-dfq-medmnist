"""Data-free, weight-only post-training quantizers for the MedMNIST compression
study. All methods operate on a float32 weight tensor and return a *reconstructed*
float32 tensor (fake-quantization), so we can measure the accuracy impact of the
quantization distortion in isolation, plus the effective bits-per-weight and the
stored size.

Methods implemented:
  - fp16            : cast to float16 and back (16 bits/weight)
  - int8_minmax     : per-output-channel symmetric INT8, round-to-nearest (data-free)
  - uniform_bbit    : per-row uniform b-bit quantizer (rotation OFF; ablation baseline)
  - uniform_full    : corrected full-cardinality symmetric mid-rise baseline
  - normal_codebook : Gaussian Lloyd-Max codebook without rotation
  - rotated_uniform : full-cardinality uniform grid after rotation
  - turboquant      : TurboQuant-MSE on weights -- random orthogonal rotation per
                      weight tensor + per-coordinate Lloyd-Max codebook for the
                      induced N(0,1) distribution + per-row fp16 scale.

The TurboQuant procedure follows Algorithm 1 of Zandieh et al., "TurboQuant:
Online Vector Quantization with Near-optimal Distortion Rate" (arXiv:2504.19874):
rotate the (unit-normalized) vector by a random orthogonal matrix so its
coordinates become approximately N(0, 1/d), then apply the Lloyd-Max optimal
scalar quantizer for that distribution per coordinate. We adapt it to weights by
treating each output channel (row of the reshaped weight) as the vector, storing
one fp16 scale per row, and regenerating the rotation from a stored seed (zero
rotation-storage cost).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Lloyd-Max optimal scalar quantizer codebooks (precomputed for N(0,1))


def lloyd_max_codebook(
    n_levels: int,
    *,
    dist: str = "gaussian",
    n_samples: int = 2_000_000,
    iters: int = 100,
    seed: int = 0,
) -> np.ndarray:
    """Compute Lloyd-Max optimal centroids for a 1-D distribution.

    This is the continuous k-means / Lloyd's algorithm: alternate between
    assigning samples to the nearest centroid and moving each centroid to the
    mean of its assigned samples. Returns sorted centroids of shape (n_levels,).
    """
    rng = np.random.default_rng(seed)
    if dist == "gaussian":
        samples = rng.standard_normal(n_samples)
    elif dist == "uniform":
        samples = rng.uniform(-1.0, 1.0, n_samples)
    else:
        raise ValueError(f"unknown dist {dist!r}")

    # Initialize centroids at evenly spaced quantiles so Lloyd converges fast.
    qs = (np.arange(n_levels) + 0.5) / n_levels
    centroids = np.quantile(samples, qs)

    for _ in range(iters):
        # Boundaries are midpoints between adjacent centroids.
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0
        idx = np.searchsorted(boundaries, samples)
        new = centroids.copy()
        for k in range(n_levels):
            mask = idx == k
            if mask.any():
                new[k] = samples[mask].mean()
        if np.allclose(new, centroids, atol=1e-7):
            centroids = new
            break
        centroids = new
    return np.sort(centroids)


def _quantize_to_codebook(x: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Map each value of x to the nearest codebook centroid; return indices."""
    boundaries = (codebook[:-1] + codebook[1:]) / 2.0
    return np.searchsorted(boundaries, x).astype(np.int64)


def _quantize_symmetric_midrise(rows: np.ndarray, bits: int):
    """Quantize rows to a full ``2**bits`` symmetric mid-rise grid.

    An even-cardinality grid cannot contain both exact zero and symmetric
    endpoints.  We retain all bit patterns by using the symmetric levels
    ``{-(L-1), -(L-3), ..., L-3, L-1} * scale``.  Exact zeros map to the
    smallest positive level; nonzero inputs satisfy Q(-x) = -Q(x).
    """
    if bits < 1:
        raise ValueError("bits must be positive")
    levels = 2 ** bits
    max_code = levels - 1
    amax = np.maximum(np.abs(rows).max(axis=1, keepdims=True), 1e-12)
    scale = amax / max_code
    magnitude = np.minimum(
        2.0 * np.floor(np.abs(rows) / (2.0 * scale)) + 1.0,
        max_code,
    )
    signed_code = np.where(rows < 0.0, -magnitude, magnitude)
    return signed_code * scale, scale


# Random orthogonal rotation (Algorithm 1, line 2: QR of a Gaussian matrix)


def random_orthogonal(d: int, seed: int) -> np.ndarray:
    """Random d x d orthogonal matrix via QR of an i.i.d. Gaussian matrix.

    The sign correction on R's diagonal makes the result Haar-distributed.
    """
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((d, d))
    q, r = np.linalg.qr(a)
    q *= np.sign(np.diag(r))
    return q


# Result container


@dataclass(frozen=True)
class QuantResult:
    """Reconstructed weights plus accounting for the compressed representation."""

    weight: np.ndarray          # reconstructed float32 weights, original shape
    bits_per_weight: float      # effective stored bits per scalar weight
    method: str


def _reshape_rows(w: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Reshape a weight tensor to (out_channels, fan_in) and return orig shape."""
    shape = w.shape
    out = shape[0]
    rows = w.reshape(out, -1).astype(np.float64)
    return rows, shape


# Quantizers


def quant_fp16(w: np.ndarray) -> QuantResult:
    rec = w.astype(np.float16).astype(np.float32)
    return QuantResult(rec, 16.0, "fp16")


def quant_int8_minmax(w: np.ndarray) -> QuantResult:
    """Per-output-channel symmetric INT8, data-free (uses weight max only)."""
    rows, shape = _reshape_rows(w)
    n = rows.shape[1]
    scale = np.maximum(np.abs(rows).max(axis=1, keepdims=True), 1e-12) / 127.0
    q = np.clip(np.round(rows / scale), -127, 127)
    rec = (q * scale).reshape(shape).astype(np.float32)
    # 8 bits/weight + one fp16 scale per row.
    bpw = 8.0 + 16.0 / n
    return QuantResult(rec, bpw, "int8_minmax")


def quant_uniform_bbit(w: np.ndarray, bits: int) -> QuantResult:
    """Per-row symmetric uniform b-bit quantizer, no rotation (ablation baseline).

    Uses the same symmetric signed convention as ``quant_int8_minmax``: with
    ``qmax = 2**(bits-1) - 1`` the step is ``amax / qmax`` and codes are clipped
    to the symmetric range ``[-qmax, qmax]``. This costs one code relative to the
    full ``2**bits`` range (e.g. 4-bit uses levels -7..7, not -7..8) but keeps
    the quantizer unbiased: ``+amax`` and ``-amax`` map to ``+qmax`` and ``-qmax``
    symmetrically. An asymmetric ``[-(2**(b-1)-1), 2**(b-1)]`` range would
    systematically overshoot positive weights and clip negative ones, which would
    make the uniform baseline an unfair comparison against the rotation method.
    """
    rows, shape = _reshape_rows(w)
    n = rows.shape[1]
    qmax = 2 ** (bits - 1) - 1
    amax = np.maximum(np.abs(rows).max(axis=1, keepdims=True), 1e-12)
    scale = amax / qmax
    q = np.clip(np.round(rows / scale), -qmax, qmax)
    rec = (q * scale).reshape(shape).astype(np.float32)
    bpw = float(bits) + 16.0 / n
    return QuantResult(rec, bpw, f"uniform_{bits}bit")


def quant_symmetric_full_bbit(w: np.ndarray, bits: int) -> QuantResult:
    """Per-row symmetric uniform quantization with all ``2**bits`` levels.

    This is the corrected matched-cardinality uniform baseline.  The earlier
    zero-centered baseline is retained as :func:`quant_uniform_bbit` solely to
    reproduce the archived experiment grid; it uses ``2**bits - 1`` levels.
    """
    rows, shape = _reshape_rows(w)
    n = rows.shape[1]
    rec, _ = _quantize_symmetric_midrise(rows, bits)
    return QuantResult(
        rec.reshape(shape).astype(np.float32),
        float(bits) + 16.0 / n,
        f"uniform_full_{bits}bit",
    )


def quant_normal_codebook(
    w: np.ndarray,
    bits: int,
    codebook: np.ndarray,
) -> QuantResult:
    """Apply the standard-normal Lloyd-Max codebook without rotation."""
    rows, shape = _reshape_rows(w)
    n = rows.shape[1]
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    scale = np.maximum(norms, 1e-12) / np.sqrt(n)
    indices = _quantize_to_codebook(rows / scale, codebook)
    rec = scale * codebook[indices]
    return QuantResult(
        rec.reshape(shape).astype(np.float32),
        float(bits) + 16.0 / n,
        f"normal_codebook_{bits}bit",
    )


def _rotated_uniform_once(rows: np.ndarray, bits: int, seed: int):
    n = rows.shape[1]
    rotation = random_orthogonal(n, seed=seed)
    rotated = rows @ rotation.T
    quantized, _ = _quantize_symmetric_midrise(rotated, bits)
    rec = quantized @ rotation
    err = float(np.sum((rec - rows) ** 2))
    return rec, err


def quant_rotated_uniform_bbit(
    w: np.ndarray,
    bits: int,
    *,
    seed: int = 0,
    n_rotations: int = 1,
) -> QuantResult:
    """Rotate each row, apply the full uniform grid, then invert rotation."""
    if n_rotations < 1:
        raise ValueError("n_rotations must be at least one")
    rows, shape = _reshape_rows(w)
    n = rows.shape[1]
    best_rec, best_err = None, np.inf
    for candidate in range(n_rotations):
        rec, err = _rotated_uniform_once(rows, bits, seed + candidate)
        if err < best_err:
            best_rec, best_err = rec, err
    return QuantResult(
        best_rec.reshape(shape).astype(np.float32),
        float(bits) + 16.0 / n,
        f"rotated_uniform_{bits}bit",
    )


def _turboquant_once(rows: np.ndarray, codebook: np.ndarray, seed: int):
    """One TurboQuant pass over rows; return (reconstructed_rows, sq_error)."""
    n = rows.shape[1]
    pi = random_orthogonal(n, seed=seed)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    s = np.maximum(norms, 1e-12) / np.sqrt(n)          # (out, 1)
    y = (rows / s) @ pi.T                               # rotate each row
    idx = _quantize_to_codebook(y, codebook)
    y_hat = codebook[idx]
    rec = s * (y_hat @ pi)
    err = float(np.sum((rec - rows) ** 2))
    return rec, err


def quant_turboquant(
    w: np.ndarray,
    bits: int,
    codebook: np.ndarray,
    *,
    seed: int = 0,
    n_rotations: int = 1,
) -> QuantResult:
    """TurboQuant-MSE applied to weights (random rotation + Lloyd-Max codebook).

    Each output channel (row) is treated as the vector to quantize:
      1. scale s = ||row|| / sqrt(n)  so rotated coordinates are ~ N(0, 1)
      2. rotate  y = Pi @ (row / s)   with a per-tensor random orthogonal Pi
      3. quantize each coordinate of y to the nearest Lloyd-Max centroid
      4. dequant: row_hat = s * (Pi^T @ codebook[idx])

    If n_rotations > 1, try that many rotation seeds and keep the one with the
    lowest reconstruction error. This is fully data-free (uses only the weights)
    and reduces the seed variance of a single random rotation.

    Stored: b bits/coord + one fp16 scale per row + a small rotation seed index.
    """
    rows, shape = _reshape_rows(w)
    n = rows.shape[1]

    best_rec, best_err = None, np.inf
    for k in range(n_rotations):
        rec, err = _turboquant_once(rows, codebook, seed=seed + k)
        if err < best_err:
            best_err, best_rec = err, rec

    if n_rotations < 1:
        raise ValueError("n_rotations must be at least one")
    rec = best_rec.reshape(shape).astype(np.float32)
    bpw = float(bits) + 16.0 / n                        # +seed cost is negligible
    return QuantResult(rec, bpw, f"rotated_normal_codebook_{bits}bit")
