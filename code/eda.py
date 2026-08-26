"""Exploratory data analysis for the four MedMNIST datasets used in the data-free
quantization study. Reports per-split sample counts, image geometry (rows x cols
x channels), feature dimensionality, class names and per-class counts, class
imbalance, and pixel-intensity statistics. Writes a Markdown report + JSON + a
figure (class distributions and pixel histograms).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from medmnist import INFO

DATASETS = ("dermamnist", "pneumoniamnist", "bloodmnist", "pathmnist")
SIZE = 64
DATA_DIR = Path(__file__).parent / "medmnist_data"
OUT_MD = Path(__file__).parent / "EDA.md"
OUT_JSON = Path(__file__).parent.parent / "results" / "dfq_eda.json"
OUT_FIG = Path(__file__).parent.parent / "results" / "dfq_eda.png"


def load_npz(name: str) -> dict:
    path = DATA_DIR / f"{name}_{SIZE}.npz"
    return dict(np.load(path))


def analyze(name: str) -> dict:
    info = INFO[name]
    label_map = info["label"]                      # {"0": "name", ...}
    n_classes = len(label_map)
    npz = load_npz(name)

    splits = {}
    pixel_samples = []
    for split in ("train", "val", "test"):
        imgs = npz[f"{split}_images"]
        labels = npz[f"{split}_labels"].reshape(-1)
        h, w = imgs.shape[1], imgs.shape[2]
        ch = 1 if imgs.ndim == 3 else imgs.shape[3]
        counts = Counter(int(c) for c in labels)
        splits[split] = {
            "n_samples": int(imgs.shape[0]),
            "image_h": int(h), "image_w": int(w), "channels": int(ch),
            "feature_dim": int(h * w * ch),
            "class_counts": {label_map[str(k)]: int(counts.get(k, 0))
                             for k in range(n_classes)},
        }
        # subsample pixels (normalized 0-1) for intensity stats
        flat = (imgs.astype(np.float32) / 255.0).reshape(imgs.shape[0], -1)
        idx = np.random.default_rng(0).choice(flat.shape[0],
                                              min(2000, flat.shape[0]), replace=False)
        pixel_samples.append(flat[idx].ravel())

    pix = np.concatenate(pixel_samples)
    train_counts = np.array(list(splits["train"]["class_counts"].values()))
    imbalance = float(train_counts.max() / max(train_counts.min(), 1))

    return {
        "dataset": name,
        "modality": info["task"],
        "description": info.get("description", "").split("\n")[0][:120],
        "n_classes": n_classes,
        "class_names": list(label_map.values()),
        "image_shape": f"{splits['train']['image_h']}x{splits['train']['image_w']}"
                       f"x{splits['train']['channels']}",
        "feature_dim": splits["train"]["feature_dim"],
        "total_samples": sum(s["n_samples"] for s in splits.values()),
        "splits": splits,
        "train_imbalance_ratio": round(imbalance, 2),
        "pixel_mean": round(float(pix.mean()), 4),
        "pixel_std": round(float(pix.std()), 4),
        "pixel_min": round(float(pix.min()), 4),
        "pixel_max": round(float(pix.max()), 4),
    }


def write_report(reports: list[dict]) -> None:
    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(reports, indent=2))

    lines = ["# MedMNIST EDA (data-free quantization study)\n",
             f"All datasets at {SIZE}x{SIZE} resolution (MedMNIST+ variant).\n",
             "## Summary\n",
             "| dataset | task | image (HxWxC) | feature dim | classes | "
             "total | train/val/test | imbalance |",
             "|---|---|---|---|---|---|---|---|"]
    for r in reports:
        s = r["splits"]
        lines.append(
            f"| {r['dataset']} | {r['modality']} | {r['image_shape']} | "
            f"{r['feature_dim']} | {r['n_classes']} | {r['total_samples']} | "
            f"{s['train']['n_samples']}/{s['val']['n_samples']}/{s['test']['n_samples']} | "
            f"{r['train_imbalance_ratio']}:1 |")

    lines.append("\n## Pixel intensity (normalized 0-1)\n")
    lines.append("| dataset | mean | std | min | max |")
    lines.append("|---|---|---|---|---|")
    for r in reports:
        lines.append(f"| {r['dataset']} | {r['pixel_mean']} | {r['pixel_std']} | "
                     f"{r['pixel_min']} | {r['pixel_max']} |")

    for r in reports:
        lines.append(f"\n## {r['dataset']} class distribution (train)\n")
        lines.append("| class | count |")
        lines.append("|---|---|")
        for cname, cnt in r["splits"]["train"]["class_counts"].items():
            lines.append(f"| {cname} | {cnt} |")

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_MD}\nwrote {OUT_JSON}")


def make_figure(reports: list[dict]) -> None:
    fig, axes = plt.subplots(2, len(reports), figsize=(4 * len(reports), 6),
                             squeeze=False)
    for j, r in enumerate(reports):
        counts = r["splits"]["train"]["class_counts"]
        ax = axes[0][j]
        ax.bar(range(len(counts)), list(counts.values()))
        ax.set_title(f"{r['dataset']}\n(train class counts)")
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels([n[:6] for n in counts], rotation=45, fontsize=7)

        npz = load_npz(r["dataset"])
        imgs = (npz["train_images"][:1000].astype(np.float32) / 255.0).ravel()
        ax2 = axes[1][j]
        ax2.hist(imgs, bins=50)
        ax2.set_title(f"{r['dataset']} pixel hist")
        ax2.set_xlabel("intensity")
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=150)
    print(f"wrote {OUT_FIG}")


def main() -> None:
    reports = [analyze(n) for n in DATASETS]
    write_report(reports)
    make_figure(reports)
    # console summary
    for r in reports:
        s = r["splits"]
        print(f"{r['dataset']:16s} {r['image_shape']:12s} feat={r['feature_dim']:5d} "
              f"cls={r['n_classes']} N={r['total_samples']:6d} "
              f"({s['train']['n_samples']}/{s['val']['n_samples']}/{s['test']['n_samples']}) "
              f"imbalance={r['train_imbalance_ratio']}:1")


if __name__ == "__main__":
    main()
