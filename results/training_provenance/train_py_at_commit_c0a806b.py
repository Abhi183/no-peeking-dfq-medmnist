"""
dfq/train.py
------------
Train an FP32 MobileNetV3-Small baseline on a MedMNIST dataset and save the
checkpoint plus clean test metrics. One checkpoint per (dataset, seed).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score

from data import load_dataset
from model import build_model

CKPT_DIR = Path(__file__).parent / "checkpoints"


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model: nn.Module, loader, device, n_classes: int) -> dict:
    model.eval()
    all_logits, all_y = [], []
    for x, y in loader:
        logits = model(x.to(device)).cpu()
        all_logits.append(logits)
        all_y.append(y)
    logits = torch.cat(all_logits)
    y = torch.cat(all_y).numpy()
    probs = torch.softmax(logits, dim=1).numpy()
    preds = probs.argmax(axis=1)
    acc = float((preds == y).mean())
    f1 = float(f1_score(y, preds, average="macro"))
    try:
        if n_classes == 2:
            auc = float(roc_auc_score(y, probs[:, 1]))
        else:
            auc = float(roc_auc_score(y, probs, multi_class="ovr", average="macro"))
    except ValueError:
        auc = float("nan")
    return {"acc": acc, "auc": auc, "f1_macro": f1}


def train_one(dataset: str, seed: int, epochs: int, size: int, lr: float) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = get_device()
    bundle = load_dataset(dataset, size=size)
    model = build_model(bundle.n_classes, bundle.n_channels, pretrained=True).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()

    t0 = time.time()
    best_auc, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        for x, y in bundle.train:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
        sched.step()
        val = evaluate(model, bundle.val, device, bundle.n_classes)
        if val["auc"] >= best_auc:
            best_auc = val["auc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[{dataset} s{seed}] epoch {ep+1}/{epochs} "
              f"val_acc={val['acc']:.4f} val_auc={val['auc']:.4f}")

    model.load_state_dict(best_state)
    test = evaluate(model, bundle.test, device, bundle.n_classes)
    secs = time.time() - t0

    CKPT_DIR.mkdir(exist_ok=True)
    ckpt = CKPT_DIR / f"{dataset}_seed{seed}.pt"
    torch.save({"state_dict": best_state, "n_classes": bundle.n_classes,
                "n_channels": bundle.n_channels, "size": size,
                "dataset": dataset, "seed": seed, "test": test}, ckpt)
    print(f"[{dataset} s{seed}] TEST {test} | {secs:.0f}s -> {ckpt.name}")
    return {"dataset": dataset, "seed": seed, "train_secs": secs, **test}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["dermamnist"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    rows = []
    for ds in args.datasets:
        for seed in args.seeds:
            rows.append(train_one(ds, seed, args.epochs, args.size, args.lr))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
