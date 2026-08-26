"""Build paper artifacts from the leakage-free grid (dfq_final_grid.csv):
config is chosen per (dataset, bit, family) by mean VALIDATION AUC, then its
TEST metrics are reported. Emits the LaTeX low-bit table (AUC/Acc/F1) and
regenerates fig2 (4-bit bars), fig3 (AUC vs bits), fig4 (rotation selection).
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RES = Path(__file__).parent.parent / "results"
FIG = Path(__file__).parent / "paper" / "figures"
GRID = pd.read_csv(RES / "dfq_final_grid.csv")
DATASETS = ["dermamnist", "pneumoniamnist", "bloodmnist", "pathmnist"]
NICE = {"dermamnist": "DermaMNIST", "pneumoniamnist": "PneumoniaMNIST",
        "bloodmnist": "BloodMNIST", "pathmnist": "PathMNIST"}
UCFG = ["naive", "policy"]
TCFG = ["naive", "policy1", "policy8"]
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})

val = GRID[GRID.split == "val"]; test = GRID[GRID.split == "test"]


def pick(ds, bit, family):
    """Return the test rows (per seed) for the val-selected config."""
    cfgs = ([f"uniform_{bit}bit_{t}" for t in UCFG] if family == "uniform"
            else [f"turboquant_{bit}bit_{t}" for t in TCFG])
    best, bv = None, -1
    for m in cfgs:
        v = val[(val.dataset == ds) & (val.method == m)].auc.mean()
        if v > bv:
            bv, best = v, m
    return test[(test.dataset == ds) & (test.method == best)], best


def ref(ds, m, col):
    s = test[(test.dataset == ds) & (test.method == m)][col]
    return s.mean(), (s.std() if len(s) > 1 else 0.0)


def latex_table():
    rows = []
    for ds in DATASETS:
        for i, b in enumerate((4, 3, 2)):
            u, _ = pick(ds, b, "uniform"); t, _ = pick(ds, b, "turboquant")
            ua, us = u.auc.mean(), u.auc.std(); ta, ts = t.auc.mean(), t.auc.std()
            ub = "\\textbf{" if ua >= ta else ""; tb = "\\textbf{" if ta > ua else ""
            ue = "}" if ub else ""; te = "}" if tb else ""
            head = f"\\multirow{{3}}{{*}}{{{NICE[ds]}}}\n" if i == 0 else ""
            rows.append(
                f"{head} & {b} & {ub}{ua:.3f}{ue}$\\pm${us:.3f} & {u.acc.mean():.2f} & {u.f1_macro.mean():.2f} "
                f"& {tb}{ta:.3f}{te}$\\pm${ts:.3f} & {t.acc.mean():.2f} & {t.f1_macro.mean():.2f} \\\\")
    print("\n".join(rows))


def fig_main_4bit():
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(DATASETS)); w = 0.2
    series = {"FP32": [], "INT8": [], "uniform 4b": [], "TurboQuant 4b": []}
    errs = {k: [] for k in series}
    for ds in DATASETS:
        m, s = ref(ds, "fp32", "auc"); series["FP32"].append(m); errs["FP32"].append(s)
        m, s = ref(ds, "int8_minmax", "auc"); series["INT8"].append(m); errs["INT8"].append(s)
        u, _ = pick(ds, 4, "uniform"); series["uniform 4b"].append(u.auc.mean()); errs["uniform 4b"].append(u.auc.std())
        t, _ = pick(ds, 4, "turboquant"); series["TurboQuant 4b"].append(t.auc.mean()); errs["TurboQuant 4b"].append(t.auc.std())
    for i, k in enumerate(series):
        ax.bar(x + (i - 1.5) * w, series[k], w, yerr=errs[k], capsize=3, label=k)
    ax.set_xticks(x); ax.set_xticklabels([NICE[d] for d in DATASETS], fontsize=8)
    ax.set_ylabel("test AUC (mean $\\pm$ std, 3 seeds)"); ax.set_ylim(0.4, 1.02)
    ax.set_title("Data-free quantization at 4 bits/weight (validation-selected config)")
    ax.legend(ncol=4, fontsize=8, loc="lower center")
    fig.tight_layout(); fig.savefig(FIG / "fig2_main_4bit.png", dpi=200); print("fig2 done")


def fig_auc_vs_bits():
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.8), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        for fam, mk in [("turboquant", "o"), ("uniform", "s")]:
            xs, ys, es = [], [], []
            for b in (2, 3, 4):
                r, _ = pick(ds, b, fam)
                xs.append(r.bpw.mean()); ys.append(r.auc.mean()); es.append(r.auc.std())
            ax.errorbar(xs, ys, yerr=es, marker=mk, capsize=2,
                        label=("TurboQuant" if fam == "turboquant" else "uniform"))
        ax.axhline(ref(ds, "fp32", "auc")[0], color="gray", ls="--", lw=1, label="FP32")
        ax.scatter([8.11], [ref(ds, "int8_minmax", "auc")[0]], color="green", marker="*", s=80, zorder=5, label="INT8")
        ax.set_title(NICE[ds], fontsize=9); ax.set_xlabel("bits / weight")
    axes[0].set_ylabel("test AUC"); axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "fig3_auc_vs_bits.png", dpi=200); print("fig3 done")


def fig_rotation():
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    x = np.arange(len(DATASETS)); w = 0.35
    def series(method):
        m, e = [], []
        for ds in DATASETS:
            s = test[(test.dataset == ds) & (test.method == method)].auc
            m.append(s.mean()); e.append(s.std())
        return m, e
    m1, e1 = series("turboquant_4bit_policy1"); m8, e8 = series("turboquant_4bit_policy8")
    ax.bar(x - w/2, m1, w, yerr=e1, capsize=3, label="1 rotation (policy)")
    ax.bar(x + w/2, m8, w, yerr=e8, capsize=3, label="best of 8 (policy)")
    ax.set_xticks(x); ax.set_xticklabels([NICE[d] for d in DATASETS], fontsize=8)
    ax.set_ylabel("TurboQuant 4-bit test AUC"); ax.set_ylim(0.5, 1.02)
    ax.set_title("Effect of data-free rotation selection"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig4_rotation.png", dpi=200); print("fig4 done")


if __name__ == "__main__":
    print("=== LaTeX low-bit table rows ===")
    latex_table()
    print("\n=== INT8/FP32 per-dataset (test) ===")
    for ds in DATASETS:
        print(f"{ds}: fp32 auc={ref(ds,'fp32','auc')[0]:.3f} acc={ref(ds,'fp32','acc')[0]:.3f} | "
              f"int8 auc={ref(ds,'int8_minmax','auc')[0]:.3f} acc={ref(ds,'int8_minmax','acc')[0]:.3f}")
    fig_main_4bit(); fig_auc_vs_bits(); fig_rotation()
