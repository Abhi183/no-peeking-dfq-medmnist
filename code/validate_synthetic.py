"""Evaluate the matched 2x2 rotation/grid ablation on synthetic weight matrices.

Reports normalized MSE  ||W - W_hat||^2 / ||W||^2  per method and bit-width, for
both a Gaussian weight matrix and a heavy-tailed (Student-t) one that better
matches real trained weights.
"""

from __future__ import annotations

import numpy as np

from quantizers import (
    lloyd_max_codebook,
    quant_normal_codebook,
    quant_rotated_uniform_bbit,
    quant_symmetric_full_bbit,
    quant_turboquant,
)


def nmse(w: np.ndarray, w_hat: np.ndarray) -> float:
    return float(np.sum((w - w_hat) ** 2) / np.sum(w ** 2))


def main() -> None:
    rng = np.random.default_rng(2026)
    out, fan_in = 256, 512

    matrices = {
        "gaussian": rng.standard_normal((out, fan_in)).astype(np.float32),
        "student_t(df=3)": (rng.standard_t(3, size=(out, fan_in))).astype(np.float32),
    }

    bit_widths = [2, 3, 4]
    codebooks = {b: lloyd_max_codebook(2 ** b) for b in bit_widths}

    print(f"{'matrix':16s} {'bits':>4s} {'uniform':>10s} {'rot+uniform':>12s} "
          f"{'gaussian':>10s} {'rot+gaussian':>13s}")
    print("-" * 82)
    for name, w in matrices.items():
        for b in bit_widths:
            codebook = codebooks[b]
            results = [
                quant_symmetric_full_bbit(w, b),
                quant_rotated_uniform_bbit(w, b, seed=17),
                quant_normal_codebook(w, b, codebook),
                quant_turboquant(w, b, codebook, seed=17),
            ]
            errors = [nmse(w, result.weight) for result in results]
            print(
                f"{name:16s} {b:>4d} "
                + " ".join(f"{value:>10.5f}" for value in errors)
            )
        print("-" * 82)


if __name__ == "__main__":
    main()
