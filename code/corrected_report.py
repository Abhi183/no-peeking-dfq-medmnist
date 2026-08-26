"""Generate tables and figures for the corrected No Peeking protocol."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURE_DIR = Path(__file__).parent.parent / "paper" / "figures"
FIXED_RESULTS = RESULTS_DIR / "dfq_corrected_fixed.csv"
FACTORIAL_RESULTS = RESULTS_DIR / "dfq_rotation_factorial.csv"
SUMMARY_PATH = RESULTS_DIR / "dfq_corrected_summary.md"

DATASETS = ("dermamnist", "pneumoniamnist", "bloodmnist", "pathmnist")
DISPLAY = {
    "dermamnist": "DermaMNIST",
    "pneumoniamnist": "PneumoniaMNIST",
    "bloodmnist": "BloodMNIST",
    "pathmnist": "PathMNIST",
}
METHODS_2X2 = (
    "uniform_full",
    "rotated_uniform",
    "normal_codebook",
    "rotated_normal_codebook",
)
SYNTHETIC_DATA_SEED = 2026
SYNTHETIC_ROTATION_SEED = 17
METHOD_LABELS = {
    "uniform_full": "uniform",
    "rotated_uniform": "rotation + uniform",
    "normal_codebook": "Gaussian codebook",
    "rotated_normal_codebook": "rotation + Gaussian codebook",
}


def summarize_metric(values: pd.Series) -> tuple[float, float]:
    return float(values.mean()), float(values.std(ddof=1))


def paired_deltas(
    frame: pd.DataFrame,
    left: str,
    right: str,
    *,
    metric: str,
    bits: int | None = None,
) -> pd.Series:
    subset = frame[frame.method.isin([left, right])].copy()
    if bits is not None and "bits" in subset:
        subset = subset[subset.bits == bits]
    index = ["dataset", "model_seed"]
    if "rotation_seed" in subset and subset.rotation_seed.nunique() > 1:
        index.append("rotation_seed")
    paired = subset.pivot(index=index, columns="method", values=metric)
    paired = paired.dropna(subset=[left, right]).sort_index()
    return (paired[right] - paired[left]).reset_index(drop=True)


def _method_rows(frame, bits, methods):
    return frame[(frame.bits == bits) & frame.method.isin(methods)]


def _write_summary(fixed: pd.DataFrame, factorial: pd.DataFrame) -> None:
    lines = [
        "# Corrected fixed-protocol results",
        "",
        "All low-bit configurations are fixed before evaluation. The uniform "
        "baseline and Gaussian codebook each use 2^b reconstruction levels. "
        "Rotation seed 0 is shared across training seeds and is independent of "
        "the checkpoint seed.",
        "",
        "## Four-bit primary comparison",
        "",
        "| dataset | uniform AUC | rotation + Gaussian AUC | paired delta |",
        "|---|---:|---:|---:|",
    ]
    primary = _method_rows(
        fixed,
        4,
        ("uniform_full", "rotated_normal_codebook"),
    )
    for dataset in DATASETS:
        block = primary[primary.dataset == dataset]
        uniform = block[block.method == "uniform_full"].auc
        rotated = block[block.method == "rotated_normal_codebook"].auc
        um, us = summarize_metric(uniform)
        rm, rs = summarize_metric(rotated)
        delta = paired_deltas(
            block,
            "uniform_full",
            "rotated_normal_codebook",
            metric="auc",
            bits=4,
        )
        dm, ds = summarize_metric(delta)
        lines.append(
            f"| {DISPLAY[dataset]} | {um:.3f} ± {us:.3f} | "
            f"{rm:.3f} ± {rs:.3f} | {dm:+.3f} ± {ds:.3f} |"
        )

    lines += [
        "",
        "## Four-bit 2x2 ablation",
        "",
        "| dataset | uniform | rot. + uniform | Gaussian | rot. + Gaussian |",
        "|---|---:|---:|---:|---:|",
    ]
    ablation = _method_rows(fixed, 4, METHODS_2X2)
    for dataset in DATASETS:
        cells = []
        for method in METHODS_2X2:
            values = ablation[
                (ablation.dataset == dataset) & (ablation.method == method)
            ].auc
            mean, std = summarize_metric(values)
            cells.append(f"{mean:.3f} ± {std:.3f}")
        lines.append(f"| {DISPLAY[dataset]} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Rotation-seed variance at four bits",
        "",
        "| dataset | mean AUC | mean within-model SD | between-model SD |",
        "|---|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        block = factorial[factorial.dataset == dataset]
        within = block.groupby("model_seed").auc.std(ddof=1)
        model_means = block.groupby("model_seed").auc.mean()
        lines.append(
            f"| {DISPLAY[dataset]} | {block.auc.mean():.3f} | "
            f"{within.mean():.3f} | {model_means.std(ddof=1):.3f} |"
        )
    SUMMARY_PATH.write_text("\n".join(lines) + "\n")


def _plot_synthetic() -> None:
    from quantizers import (
        lloyd_max_codebook,
        quant_normal_codebook,
        quant_rotated_uniform_bbit,
        quant_symmetric_full_bbit,
        quant_turboquant,
    )

    rng = np.random.default_rng(SYNTHETIC_DATA_SEED)
    matrices = {
        "Gaussian weights": rng.standard_normal((128, 256)).astype(np.float32),
        "Heavy-tailed weights": rng.standard_t(3, (128, 256)).astype(np.float32),
    }
    bits = (2, 3, 4)
    codebooks = {bit: lloyd_max_codebook(2**bit) for bit in bits}
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), sharey=True)

    def nmse(weight, reconstruction):
        return float(
            np.sum((weight - reconstruction) ** 2) / np.sum(weight**2)
        )

    for ax, (title, weight) in zip(axes, matrices.items()):
        series = {method: [] for method in METHODS_2X2}
        for bit in bits:
            codebook = codebooks[bit]
            results = {
                "uniform_full": quant_symmetric_full_bbit(weight, bit),
                "rotated_uniform": quant_rotated_uniform_bbit(
                    weight,
                    bit,
                    seed=SYNTHETIC_ROTATION_SEED,
                ),
                "normal_codebook": quant_normal_codebook(
                    weight,
                    bit,
                    codebook,
                ),
                "rotated_normal_codebook": quant_turboquant(
                    weight,
                    bit,
                    codebook,
                    seed=SYNTHETIC_ROTATION_SEED,
                ),
            }
            for method, result in results.items():
                series[method].append(nmse(weight, result.weight))
        for method, values in series.items():
            ax.plot(bits, values, marker="o", label=METHOD_LABELS[method])
        ax.set_title(title)
        ax.set_xlabel("bits per weight")
        ax.set_xticks(bits)
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("reconstruction NMSE")
    axes[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig1_corrected_synthetic.png", dpi=240)
    plt.close(fig)


def _plot_ablation(fixed: pd.DataFrame) -> None:
    block = _method_rows(fixed, 4, METHODS_2X2)
    x = np.arange(len(DATASETS))
    width = 0.19
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    for offset, method in enumerate(METHODS_2X2):
        means, errors = [], []
        for dataset in DATASETS:
            values = block[
                (block.dataset == dataset) & (block.method == method)
            ].auc
            mean, std = summarize_metric(values)
            means.append(mean)
            errors.append(std)
        ax.bar(
            x + (offset - 1.5) * width,
            means,
            width,
            yerr=errors,
            capsize=2,
            label=METHOD_LABELS[method],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[item] for item in DATASETS], fontsize=8)
    ax.set_ylabel("test AUC")
    ax.set_ylim(0.35, 1.02)
    ax.legend(fontsize=7, ncol=2, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig2_corrected_ablation.png", dpi=240)
    plt.close(fig)


def _plot_bits(fixed: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 2.7), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        for method, marker in (
            ("uniform_full", "s"),
            ("rotated_normal_codebook", "o"),
        ):
            means, errors = [], []
            for bit in (2, 3, 4):
                values = fixed[
                    (fixed.dataset == dataset)
                    & (fixed.method == method)
                    & (fixed.bits == bit)
                ].auc
                mean, std = summarize_metric(values)
                means.append(mean)
                errors.append(std)
            ax.errorbar(
                (2, 3, 4),
                means,
                yerr=errors,
                marker=marker,
                capsize=2,
                label=METHOD_LABELS[method],
            )
        fp32 = fixed[
            (fixed.dataset == dataset) & (fixed.method == "fp32")
        ].auc.mean()
        int8 = fixed[
            (fixed.dataset == dataset) & (fixed.method == "int8_minmax")
        ].auc.mean()
        ax.axhline(fp32, color="gray", linestyle="--", linewidth=1)
        ax.scatter([8.0], [int8], color="green", marker="*", s=55)
        ax.set_title(DISPLAY[dataset], fontsize=9)
        ax.set_xlabel("bits per weight")
        ax.set_xticks((2, 3, 4, 8))
        ax.set_xlim(1.7, 8.3)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("test AUC")
    axes[0].legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig3_corrected_bits.png", dpi=240)
    plt.close(fig)


def _plot_factorial(factorial: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 2.7), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        block = factorial[factorial.dataset == dataset]
        for model_seed, values in block.groupby("model_seed"):
            values = values.sort_values("rotation_seed")
            ax.plot(
                values.rotation_seed,
                values.auc,
                marker="o",
                label=f"model seed {model_seed}",
            )
        ax.set_title(DISPLAY[dataset], fontsize=9)
        ax.set_xlabel("rotation seed")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("4-bit test AUC")
    axes[-1].legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig4_rotation_factorial.png", dpi=240)
    plt.close(fig)


def generate() -> None:
    fixed = pd.read_csv(FIXED_RESULTS)
    factorial = pd.read_csv(FACTORIAL_RESULTS)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    _write_summary(fixed, factorial)
    _plot_synthetic()
    _plot_ablation(fixed)
    _plot_bits(fixed)
    _plot_factorial(factorial)
    print(f"wrote {SUMMARY_PATH}")
    print(f"wrote corrected figures to {FIGURE_DIR}")


if __name__ == "__main__":
    generate()
