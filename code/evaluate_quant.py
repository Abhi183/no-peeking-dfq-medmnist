"""Load a trained FP32 checkpoint, apply each data-free weight quantizer to every
Conv2d / Linear weight tensor (fake-quantization: weights are reconstructed in
fp32 so we isolate the quantization distortion's effect on accuracy), and report
test accuracy / AUC / F1, effective bits-per-weight, and model size per method.

Two modes:
  - naive  (default): quantize every eligible tensor uniformly at the target bits.
  - policy (--policy): mixed precision -- keep the first conv and final classifier
    at INT8, and skip rotation on tensors whose fan-in < MIN_ROTATE_DIM (the
    Gaussian limit that TurboQuant relies on does not hold for tiny rows such as
    depthwise 3x3 kernels, fan-in 9), falling back to uniform there.

Writes one tidy CSV row per (dataset, seed, method).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from data import load_dataset
from model import build_model
from quantizers import (
    lloyd_max_codebook,
    quant_fp16,
    quant_int8_minmax,
    quant_turboquant,
    quant_uniform_bbit,
)
from train import evaluate, get_device

CKPT_DIR = Path(__file__).parent / "checkpoints"
RESULTS_NAIVE = Path(__file__).parent.parent / "results" / "dfq_medmnist.csv"
RESULTS_POLICY = Path(__file__).parent.parent / "results" / "dfq_medmnist_policy.csv"

# Quantize weights of these layer types; skip biases, BN, and tiny tensors.
QUANT_TYPES = (nn.Conv2d, nn.Linear)
MIN_PARAMS = 256
# Below this fan-in the rotated coordinates are too far from N(0,1) for the
# Lloyd-Max codebook to help, so the policy mode skips rotation there.
MIN_ROTATE_DIM = 64
# Number of random rotations to try per tensor; keep the lowest-error one.
# Data-free (uses only weights). 1 = original single-rotation behaviour.
N_ROTATIONS = 8


def _quantizable_names(model: nn.Module) -> list[str]:
    names = []
    for mod_name, mod in model.named_modules():
        if isinstance(mod, QUANT_TYPES):
            if mod.weight.numel() >= MIN_PARAMS:
                names.append(f"{mod_name}.weight")
    return names


def _fan_in(w: np.ndarray) -> int:
    return int(np.prod(w.shape[1:]))


def _quantize_tensor(name, w, family, b, *, protected, codebook, seed, policy):
    """Return a QuantResult for one tensor under the chosen mode/family."""
    if policy and name in protected:
        return quant_int8_minmax(w)                      # keep sensitive layers at 8-bit
    if family == "uniform":
        return quant_uniform_bbit(w, b)
    # family == "turboquant"
    if policy and _fan_in(w) < MIN_ROTATE_DIM:
        return quant_uniform_bbit(w, b)                  # rotation skip on narrow rows
    return quant_turboquant(w, b, codebook, seed=seed, n_rotations=N_ROTATIONS)


def _apply(state, names, family, b, *, protected, codebook, seed, policy):
    """Quantize the named tensors; return new state, per-tensor bpw map, mean bpw."""
    new_state = {k: v.clone() for k, v in state.items()}
    bpw_map, bpws, counts = {}, [], []
    for name in names:
        w = state[name].cpu().numpy()
        res = _quantize_tensor(name, w, family, b, protected=protected,
                               codebook=codebook, seed=seed, policy=policy)
        new_state[name] = torch.from_numpy(res.weight)
        bpw_map[name] = res.bits_per_weight
        bpws.append(res.bits_per_weight)
        counts.append(w.size)
    return new_state, bpw_map, float(np.average(bpws, weights=counts))


def _model_size_mb(model, names, bpw_map) -> float:
    bits, qset = 0, set(names)
    for n, p in model.named_parameters():
        bits += p.numel() * (bpw_map.get(n, 16.0) if n in qset else 16)
    return bits / 8 / 1e6


def evaluate_checkpoint(ckpt_path, *, bit_widths=(2, 3, 4), policy=False) -> list[dict]:
    device = get_device()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    dataset, seed = ckpt["dataset"], ckpt["seed"]
    bundle = load_dataset(dataset, size=ckpt["size"])
    model = build_model(ckpt["n_classes"], ckpt["n_channels"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    names = _quantizable_names(model)
    protected = {names[0], names[-1]} if policy else set()  # first conv, final fc
    base_state = {k: v.clone() for k, v in ckpt["state_dict"].items()}
    codebooks = {b: lloyd_max_codebook(2 ** b) for b in bit_widths}

    rows = []

    def run(method, new_state, bpw_map, mean_bpw):
        model.load_state_dict(new_state)
        model.to(device)
        m = evaluate(model, bundle.test, device, bundle.n_classes)
        size = _model_size_mb(model, names, bpw_map)
        rows.append({"dataset": dataset, "seed": seed, "method": method,
                     "mode": "policy" if policy else "naive",
                     "bits_per_weight": round(mean_bpw, 3), "size_mb": round(size, 4),
                     **{k: round(v, 4) for k, v in m.items()}})
        print(f"[{dataset} s{seed}] {method:18s}({'policy' if policy else 'naive'}) "
              f"acc={m['acc']:.4f} auc={m['auc']:.4f} f1={m['f1_macro']:.4f} "
              f"bpw={mean_bpw:.2f} size={size:.2f}MB")

    # References (mode-independent).
    run("fp32", base_state, {n: 32.0 for n in names}, 32.0)
    for ref_name, fn in [("fp16", quant_fp16), ("int8_minmax", quant_int8_minmax)]:
        ns = {k: v.clone() for k, v in base_state.items()}
        bm = {}
        for n in names:
            r = fn(base_state[n].cpu().numpy())
            ns[n] = torch.from_numpy(r.weight); bm[n] = r.bits_per_weight
        mean = float(np.average(list(bm.values()),
                                weights=[base_state[n].numel() for n in names]))
        run(ref_name, ns, bm, mean)

    # Swept families.
    for b in bit_widths:
        for family in ("uniform", "turboquant"):
            ns, bm, mean = _apply(base_state, names, family, b, protected=protected,
                                  codebook=codebooks[b], seed=seed, policy=policy)
            run(f"{family}_{b}bit", ns, bm, mean)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="*", default=None)
    ap.add_argument("--policy", action="store_true",
                    help="mixed-precision policy (protect first/last, skip narrow rotation)")
    args = ap.parse_args()

    paths = ([Path(c) for c in args.ckpts] if args.ckpts
             else sorted(CKPT_DIR.glob("*.pt")))
    all_rows = []
    for p in paths:
        all_rows.extend(evaluate_checkpoint(p, policy=args.policy))

    out = RESULTS_POLICY if args.policy else RESULTS_NAIVE
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nwrote {len(all_rows)} rows -> {out}")


if __name__ == "__main__":
    main()
