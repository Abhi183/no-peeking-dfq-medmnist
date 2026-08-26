"""MedMNIST v2 data loaders for the data-free quantization study. Small, standardized,
MIT-licensed medical image datasets that download automatically and run locally.

We use four modalities to show the quantization method generalizes:
  - dermamnist     : dermatoscopy, 7 classes, RGB
  - pneumoniamnist : chest X-ray, 2 classes, grayscale
  - bloodmnist     : blood cells, 8 classes, RGB
  - pathmnist      : colon pathology, 9 classes, RGB
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import medmnist
from medmnist import INFO

DATASETS = ("dermamnist", "pneumoniamnist", "bloodmnist", "pathmnist")


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    n_classes: int
    n_channels: int
    train: DataLoader
    val: DataLoader
    test: DataLoader


def _to_tensors(imgs: np.ndarray, labels: np.ndarray, n_channels: int):
    """MedMNIST images come as (N,H,W) gray or (N,H,W,3) RGB uint8."""
    x = imgs.astype(np.float32) / 255.0
    if x.ndim == 3:                      # grayscale -> (N,1,H,W)
        x = x[:, None, :, :]
    else:                                # RGB -> (N,3,H,W)
        x = np.transpose(x, (0, 3, 1, 2))
    # ImageNet-style normalization (single channel uses the mean of the 3).
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    if n_channels == 1:
        mean, std = mean.mean(keepdims=True), std.mean(keepdims=True)
    x = (x - mean[None, :, None, None]) / std[None, :, None, None]
    y = labels.astype(np.int64).reshape(-1)
    return torch.from_numpy(x), torch.from_numpy(y)


def load_dataset(
    name: str,
    *,
    size: int = 64,
    batch_size: int = 128,
    root: str | None = None,
) -> DatasetBundle:
    """Download (if needed) and wrap a MedMNIST dataset as DataLoaders."""
    if name not in INFO:
        raise ValueError(f"unknown MedMNIST dataset {name!r}")
    info = INFO[name]
    n_channels = info["n_channels"]
    n_classes = len(info["label"])
    DataClass = getattr(medmnist, info["python_class"])

    if root is None:
        root = str(Path(__file__).parent / "medmnist_data")
    Path(root).mkdir(parents=True, exist_ok=True)

    splits = {}
    for split in ("train", "val", "test"):
        ds = DataClass(split=split, download=True, size=size, root=root)
        x, y = _to_tensors(ds.imgs, ds.labels, n_channels)
        loader = DataLoader(
            TensorDataset(x, y),
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,
        )
        splits[split] = loader

    return DatasetBundle(
        name=name,
        n_classes=n_classes,
        n_channels=n_channels,
        train=splits["train"],
        val=splits["val"],
        test=splits["test"],
    )
