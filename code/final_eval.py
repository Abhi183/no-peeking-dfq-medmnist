"""Leakage-free evaluation: select the quantization configuration (layer-precision
policy on/off, and number of rotations for the TurboQuant-MSE-style method) using
ONLY the validation split, then report the selected configuration's test metrics.
Every config is scored on the full val and full test sets (no subsampling).

Outputs:
  results/dfq_final_grid.csv      : every config on both splits, per seed
  results/dfq_final_selected.csv  : val-selected config + its test metrics
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

# NOTE: torch and the model/data/quantizer stack are imported lazily inside
# run_grid() so that read-only reporting (--select-only, which only needs pandas
# and the existing grid CSV) works in lightweight environments without torch.

CKPT_DIR = Path(__file__).parent / "checkpoints"
RES = Path(__file__).parent.parent / "results"
GRID = RES / "dfq_final_grid.csv"
SEL = RES / "dfq_final_selected.csv"
BITS = (2, 3, 4)

# TurboQuant configs to choose among on validation: (tag, n_rotations, policy)
TURBO_CFGS = [("naive", 1, False), ("policy1", 1, True), ("policy8", 8, True)]
UNIFORM_CFGS = [("naive", False), ("policy", True)]


def _quant_tensor(name, w, family, b, *, protected, codebook, seed, policy, n_rot):
    from evaluate_quant import MIN_ROTATE_DIM, _fan_in
    from quantizers import (quant_int8_minmax, quant_turboquant,
                            quant_uniform_bbit)
    if policy and name in protected:
        return quant_int8_minmax(w)
    if family == "uniform":
        return quant_uniform_bbit(w, b)
    if policy and _fan_in(w) < MIN_ROTATE_DIM:
        return quant_uniform_bbit(w, b)
    return quant_turboquant(w, b, codebook, seed=seed, n_rotations=n_rot)


def _apply(base, names, family, b, *, protected, codebook, seed, policy, n_rot):
    import torch
    ns = {k: v.clone() for k, v in base.items()}
    bm, bpws, cnts = {}, [], []
    for name in names:
        w = base[name].cpu().numpy()
        r = _quant_tensor(name, w, family, b, protected=protected, codebook=codebook,
                          seed=seed, policy=policy, n_rot=n_rot)
        ns[name] = torch.from_numpy(r.weight)
        bm[name] = r.bits_per_weight
        bpws.append(r.bits_per_weight)
        cnts.append(w.size)
    return ns, bm, float(np.average(bpws, weights=cnts))


def run_grid() -> None:
    import torch

    from data import load_dataset
    from evaluate_quant import _model_size_mb, _quantizable_names
    from model import build_model
    from quantizers import (lloyd_max_codebook, quant_fp16,
                            quant_int8_minmax)
    from train import evaluate, get_device

    device = get_device()
    rows = []
    for p in sorted(CKPT_DIR.glob("*.pt")):
        ckpt = torch.load(p, map_location="cpu")
        ds, seed = ckpt["dataset"], ckpt["seed"]
        bundle = load_dataset(ds, size=ckpt["size"])
        model = build_model(ckpt["n_classes"], ckpt["n_channels"], pretrained=False)
        model.load_state_dict(ckpt["state_dict"]); model.to(device)
        names = _quantizable_names(model)
        protected = {names[0], names[-1]}
        base = {k: v.clone() for k, v in ckpt["state_dict"].items()}
        cbs = {b: lloyd_max_codebook(2 ** b) for b in BITS}

        def record(method, ns, bm, bpw):
            for split, loader in [("val", bundle.val), ("test", bundle.test)]:
                model.load_state_dict(ns); model.to(device)
                m = evaluate(model, loader, device, bundle.n_classes)
                size = _model_size_mb(model, names, bm) if bm else _model_size_mb(
                    model, names, {n: 32.0 for n in names})
                rows.append({"dataset": ds, "seed": seed, "method": method,
                             "split": split, "bpw": round(bpw, 3),
                             "size_mb": round(size, 4),
                             **{k: round(v, 4) for k, v in m.items()}})
            print(f"[{ds} s{seed}] {method:22s} done")

        record("fp32", base, {n: 32.0 for n in names}, 32.0)
        for nm, fn in [("fp16", quant_fp16), ("int8_minmax", quant_int8_minmax)]:
            ns = {k: v.clone() for k, v in base.items()}; bm = {}
            for n in names:
                r = fn(base[n].cpu().numpy())
                ns[n] = torch.from_numpy(r.weight); bm[n] = r.bits_per_weight
            mean = float(np.average(list(bm.values()),
                                    weights=[base[n].numel() for n in names]))
            record(nm, ns, bm, mean)
        for b in BITS:
            for tag, pol in UNIFORM_CFGS:
                ns, bm, bpw = _apply(base, names, "uniform", b, protected=protected,
                                     codebook=cbs[b], seed=seed, policy=pol, n_rot=1)
                record(f"uniform_{b}bit_{tag}", ns, bm, bpw)
            for tag, nrot, pol in TURBO_CFGS:
                ns, bm, bpw = _apply(base, names, "turboquant", b, protected=protected,
                                     codebook=cbs[b], seed=seed, policy=pol, n_rot=nrot)
                record(f"turboquant_{b}bit_{tag}", ns, bm, bpw)

    RES.mkdir(exist_ok=True)
    with open(GRID, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {GRID}")


def select() -> None:
    """For each (dataset, bit, family), pick the config with the best mean
    validation AUC across seeds, then report that config's test metrics."""
    df = pd.read_csv(GRID)
    val = df[df.split == "val"]; test = df[df.split == "test"]
    out = []
    # References: no config choice.
    for m in ["fp32", "fp16", "int8_minmax"]:
        t = test[test.method == m]
        out.append({"dataset": "ALL", "bit": "-", "family": m, "chosen": m,
                    "val_auc_mean": round(val[val.method == m].auc.mean(), 4),
                    "test_auc_mean": round(t.auc.mean(), 4),
                    "test_auc_std": round(t.auc.std(), 4),
                    "test_acc_mean": round(t.acc.mean(), 4),
                    "test_f1_mean": round(t.f1_macro.mean(), 4)})
    for ds in sorted(df.dataset.unique()):
        for b in BITS:
            for family, cfgs in [("uniform", [f"uniform_{b}bit_{t}" for t, _ in UNIFORM_CFGS]),
                                 ("turboquant", [f"turboquant_{b}bit_{t}" for t, _, _ in TURBO_CFGS])]:
                # choose by mean val AUC over seeds
                best, best_val = None, -1
                for m in cfgs:
                    vv = val[(val.dataset == ds) & (val.method == m)].auc.mean()
                    if vv > best_val:
                        best_val, best = vv, m
                t = test[(test.dataset == ds) & (test.method == best)]
                out.append({"dataset": ds, "bit": b, "family": family, "chosen": best,
                            "val_auc_mean": round(best_val, 4),
                            "test_auc_mean": round(t.auc.mean(), 4),
                            "test_auc_std": round(t.auc.std(), 4),
                            "test_acc_mean": round(t.acc.mean(), 4),
                            "test_f1_mean": round(t.f1_macro.mean(), 4)})
    pd.DataFrame(out).to_csv(SEL, index=False)
    print(f"wrote {SEL}")
    print(pd.DataFrame(out).to_string(index=False))


if __name__ == "__main__":
    import sys
    if "--select-only" in sys.argv:
        select()
    else:
        run_grid()
        select()
