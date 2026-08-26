"""Stage 1 of the leakage-free headline protocol. Score every data-free config on
the VALIDATION split only, then select, per (dataset, bit, family), the config
with the best mean validation AUC across seeds. The test split is never read
here, by construction.

Outputs:
  results/dfq_val_grid.csv          : every config, validation split, per seed
  results/dfq_selected_configs.csv  : the val-selected config per (dataset, bit, family)

Pair with test_selected_only.py (Stage 2), which evaluates ONLY these chosen
configs on the test split. Together the two scripts are the clean,
cherry-pick-proof reproduction of the headline table; final_eval.py additionally
scores the non-selected configs on test for the labeled ablation only.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

CKPT_DIR = Path(__file__).parent / "checkpoints"
RES = Path(__file__).parent.parent / "results"
VAL_GRID = RES / "dfq_val_grid.csv"
SELECTED = RES / "dfq_selected_configs.csv"


def run_val_grid() -> None:
    import torch

    from data import load_dataset
    from evaluate_quant import _model_size_mb, _quantizable_names
    from final_eval import BITS, TURBO_CFGS, UNIFORM_CFGS, _apply
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
            model.load_state_dict(ns); model.to(device)
            m = evaluate(model, bundle.val, device, bundle.n_classes)  # VAL ONLY
            size = (_model_size_mb(model, names, bm) if bm else
                    _model_size_mb(model, names, {n: 32.0 for n in names}))
            rows.append({"dataset": ds, "seed": seed, "method": method,
                         "split": "val", "bpw": round(bpw, 3),
                         "size_mb": round(size, 4),
                         **{k: round(v, 4) for k, v in m.items()}})
            print(f"[{ds} s{seed}] {method:22s} val done")

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
    with open(VAL_GRID, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {len(rows)} validation rows -> {VAL_GRID}")


def select() -> None:
    """Pick, per (dataset, bit, family), the config with the best mean validation
    AUC across seeds. Reads only the validation grid."""
    from final_eval import BITS, TURBO_CFGS, UNIFORM_CFGS

    val = pd.read_csv(VAL_GRID)
    out = []
    for ds in sorted(val.dataset.unique()):
        for b in BITS:
            families = [
                ("uniform", [f"uniform_{b}bit_{t}" for t, _ in UNIFORM_CFGS]),
                ("turboquant", [f"turboquant_{b}bit_{t}" for t, _, _ in TURBO_CFGS]),
            ]
            for family, cfgs in families:
                best, best_val = None, -1.0
                for m in cfgs:
                    vv = val[(val.dataset == ds) & (val.method == m)].auc.mean()
                    if vv > best_val:
                        best_val, best = vv, m
                out.append({"dataset": ds, "bit": b, "family": family,
                            "chosen": best, "val_auc_mean": round(best_val, 4)})
    pd.DataFrame(out).to_csv(SELECTED, index=False)
    print(f"wrote {SELECTED}")
    print(pd.DataFrame(out).to_string(index=False))


if __name__ == "__main__":
    run_val_grid()
    select()
