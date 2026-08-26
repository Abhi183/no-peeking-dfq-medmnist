"""Unit tests for the data-free weight quantizers."""

from __future__ import annotations

import numpy as np

from quantizers import (
    lloyd_max_codebook,
    quant_fp16,
    quant_int8_minmax,
    quant_normal_codebook,
    quant_rotated_uniform_bbit,
    quant_symmetric_full_bbit,
    quant_turboquant,
    quant_uniform_bbit,
    random_orthogonal,
)


def test_random_orthogonal_is_orthonormal():
    q = random_orthogonal(64, seed=1)
    assert np.allclose(q @ q.T, np.eye(64), atol=1e-10)


def test_lloyd_max_codebook_sorted_and_zero_mean():
    cb = lloyd_max_codebook(8, n_samples=200_000)
    assert cb.shape == (8,)
    assert np.all(np.diff(cb) > 0)              # strictly increasing
    assert abs(cb.mean()) < 0.05                # symmetric around 0


def test_shapes_preserved():
    w = np.random.default_rng(0).standard_normal((32, 3, 3, 3)).astype(np.float32)
    cb = lloyd_max_codebook(8)
    for res in [quant_fp16(w), quant_int8_minmax(w),
                quant_uniform_bbit(w, 4), quant_turboquant(w, 4, cb)]:
        assert res.weight.shape == w.shape
        assert res.weight.dtype == np.float32


def test_turboquant_beats_uniform_on_heavy_tailed():
    """Core paper premise: rotation+Lloyd-Max wins at low bits on heavy tails."""
    rng = np.random.default_rng(7)
    w = rng.standard_t(3, size=(128, 256)).astype(np.float32)
    cb3 = lloyd_max_codebook(8)
    u = quant_uniform_bbit(w, 3)
    t = quant_turboquant(w, 3, cb3)
    def nmse(a, b):
        return np.sum((a - b) ** 2) / np.sum(a ** 2)

    assert nmse(w, t.weight) < nmse(w, u.weight)


def test_uniform_bbit_is_symmetric():
    """Symmetric signed quantizer: +amax and -amax must map symmetrically, and
    the reconstructed code set must be symmetric about zero (no positive bias).

    Regression test for the asymmetric-range bug where 4-bit clipped to [-7, 8],
    overshooting positive weights and under-representing negative ones.
    """
    rng = np.random.default_rng(0)
    for bits in (2, 3, 4):
        qmax = 2 ** (bits - 1) - 1
        # A row whose extremes are exactly +/-A should reconstruct to +/-A.
        a = 3.0
        row = np.linspace(-a, a, 4096).reshape(1, -1).astype(np.float32)
        rec = quant_uniform_bbit(row, bits).weight
        assert np.isclose(rec.max(), a, atol=1e-5)
        assert np.isclose(rec.min(), -a, atol=1e-5)
        # Reconstructed levels are symmetric about zero.
        levels = np.unique(np.round(rec / (a / qmax)).astype(int))
        assert levels.min() == -qmax and levels.max() == qmax
        assert set(levels.tolist()) == set((-levels).tolist())
        # Exact antisymmetry: negating the input negates the reconstruction.
        # This is the structural guarantee of an unbiased symmetric quantizer
        # (the old asymmetric [-7, 8] range violated it).
        w = rng.standard_normal((16, 1024)).astype(np.float32)
        pos = quant_uniform_bbit(w, bits).weight
        neg = quant_uniform_bbit(-w, bits).weight
        assert np.allclose(neg, -pos, atol=1e-6)


def test_full_symmetric_uniform_uses_all_bit_patterns():
    """The corrected baseline must have the same number of reconstruction
    levels as the 2**b-entry Lloyd-Max codebook at the same nominal rate."""
    for bits in (2, 3, 4):
        levels = 2 ** bits
        # Avoid exact zero: an even-level symmetric mid-rise grid has no zero.
        row = np.linspace(-3.0, 3.0, 10001, dtype=np.float32)
        row = row[row != 0].reshape(1, -1)
        rec = quant_symmetric_full_bbit(row, bits).weight
        unique = np.unique(np.round(rec, decimals=6))
        assert len(unique) == levels
        assert np.isclose(unique[0], -3.0, atol=1e-5)
        assert np.isclose(unique[-1], 3.0, atol=1e-5)
        assert np.allclose(unique, -unique[::-1], atol=1e-5)


def test_full_symmetric_uniform_is_antisymmetric_away_from_zero():
    rng = np.random.default_rng(8)
    w = rng.standard_normal((8, 257)).astype(np.float32)
    w[w == 0] = 1e-6
    for bits in (2, 3, 4):
        pos = quant_symmetric_full_bbit(w, bits).weight
        neg = quant_symmetric_full_bbit(-w, bits).weight
        assert np.allclose(neg, -pos, atol=1e-6)


def test_two_by_two_ablation_preserves_shape_and_rate():
    """All four rotation/codebook cells must be executable at matched rate."""
    rng = np.random.default_rng(11)
    w = rng.standard_t(3, size=(16, 64)).astype(np.float32)
    cb = lloyd_max_codebook(16, n_samples=100_000)
    results = [
        quant_symmetric_full_bbit(w, 4),
        quant_rotated_uniform_bbit(w, 4, seed=17),
        quant_normal_codebook(w, 4, cb),
        quant_turboquant(w, 4, cb, seed=17),
    ]
    assert {r.method for r in results} == {
        "uniform_full_4bit",
        "rotated_uniform_4bit",
        "normal_codebook_4bit",
        "rotated_normal_codebook_4bit",
    }
    for result in results:
        assert result.weight.shape == w.shape
        assert result.weight.dtype == np.float32
        assert np.isclose(result.bits_per_weight, 4.25)


def test_rotation_seed_changes_rotation_not_unrotated_cells():
    rng = np.random.default_rng(12)
    w = rng.standard_t(3, size=(16, 64)).astype(np.float32)
    cb = lloyd_max_codebook(16, n_samples=100_000)

    rotated_a = quant_turboquant(w, 4, cb, seed=100).weight
    rotated_b = quant_turboquant(w, 4, cb, seed=101).weight
    assert not np.allclose(rotated_a, rotated_b)

    plain_a = quant_normal_codebook(w, 4, cb).weight
    plain_b = quant_normal_codebook(w, 4, cb).weight
    assert np.array_equal(plain_a, plain_b)


def test_bits_per_weight_accounting():
    w = np.random.default_rng(0).standard_normal((64, 512)).astype(np.float32)
    assert quant_fp16(w).bits_per_weight == 16.0
    # 4-bit + one fp16 scale per 512-wide row ~ 4.03 bpw.
    assert 4.0 < quant_uniform_bbit(w, 4).bits_per_weight < 4.1


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
