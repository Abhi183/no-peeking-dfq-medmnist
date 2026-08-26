"""Corrected, fixed-configuration evaluation for the No Peeking study.

This protocol addresses three problems in the archived experiment:

* the uniform baseline now uses all ``2**b`` reconstruction levels;
* rotation and codebook choice are crossed in a 2x2 ablation;
* rotation seeds are fixed independently of checkpoint training seeds.

No quantizer configuration is selected on validation data.  The primary methods
use one fixed layer policy, one rotation, and rotation seed zero.  A separate
factorial run varies rotation seed while holding each trained checkpoint fixed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np

CKPT_DIR = Path(__file__).parent / "checkpoints"
RESULTS_DIR = Path(__file__).parent.parent / "results"
FIXED_RESULTS = RESULTS_DIR / "dfq_corrected_fixed.csv"
FACTORIAL_RESULTS = RESULTS_DIR / "dfq_rotation_factorial.csv"

BITS = (2, 3, 4)
PRIMARY_METHODS = ("uniform_full", "rotated_normal_codebook")
ABLATION_METHODS = (
    "uniform_full",
    "rotated_uniform",
    "normal_codebook",
    "rotated_normal_codebook",
)
DEFAULT_ROTATION_SEEDS = (0, 1, 2, 3, 4)


def parse_method(method: str) -> tuple[bool, str]:
    mapping = {
        "uniform_full": (False, "uniform"),
        "rotated_uniform": (True, "uniform"),
        "normal_codebook": (False, "normal_codebook"),
        "rotated_normal_codebook": (True, "normal_codebook"),
    }
    try:
        return mapping[method]
    except KeyError as exc:
        raise ValueError(f"unknown corrected method: {method}") from exc


def make_tensor_seed(rotation_seed: int, tensor_index: int) -> int:
    """Derive a stable per-tensor seed without reading the training seed."""
    if rotation_seed < 0 or tensor_index < 0:
        raise ValueError("rotation_seed and tensor_index must be nonnegative")
    modulus = 2**32 - 5
    return int((rotation_seed * 1_000_003 + tensor_index * 97_409 + 17) % modulus)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantize_tensor(
    method,
    weight,
    bits,
    codebook,
    *,
    rotation_seed,
    tensor_index,
    allow_rotation,
):
    from quantizers import (
        quant_normal_codebook,
        quant_rotated_uniform_bbit,
        quant_symmetric_full_bbit,
        quant_turboquant,
    )

    rotated, grid = parse_method(method)
    rotated = rotated and allow_rotation
    if grid == "uniform":
        if rotated:
            return quant_rotated_uniform_bbit(
                weight,
                bits,
                seed=make_tensor_seed(rotation_seed, tensor_index),
            )
        return quant_symmetric_full_bbit(weight, bits)
    if rotated:
        return quant_turboquant(
            weight,
            bits,
            codebook,
            seed=make_tensor_seed(rotation_seed, tensor_index),
        )
    return quant_normal_codebook(weight, bits, codebook)


def _apply_fixed(
    base,
    names,
    method,
    bits,
    *,
    protected,
    codebook,
    rotation_seed,
):
    import torch

    from evaluate_quant import MIN_ROTATE_DIM, _fan_in
    from quantizers import quant_int8_minmax

    state = {key: value.clone() for key, value in base.items()}
    bpw_map, rates, counts = {}, [], []
    for tensor_index, name in enumerate(names):
        weight = base[name].cpu().numpy()
        if name in protected:
            result = quant_int8_minmax(weight)
        else:
            result = _quantize_tensor(
                method,
                weight,
                bits,
                codebook,
                rotation_seed=rotation_seed,
                tensor_index=tensor_index,
                allow_rotation=_fan_in(weight) >= MIN_ROTATE_DIM,
            )
        state[name] = torch.from_numpy(result.weight)
        bpw_map[name] = result.bits_per_weight
        rates.append(result.bits_per_weight)
        counts.append(weight.size)
    return state, bpw_map, float(np.average(rates, weights=counts))


def _load_checkpoint(path, device):
    import torch

    from data import load_dataset
    from evaluate_quant import _quantizable_names
    from model import build_model

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    # Evaluation is inference-only; a large batch substantially reduces CPU
    # overhead while preserving logits and metrics.
    bundle = load_dataset(
        checkpoint["dataset"],
        size=checkpoint["size"],
        batch_size=1024,
    )
    model = build_model(
        checkpoint["n_classes"],
        checkpoint["n_channels"],
        pretrained=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    names = _quantizable_names(model)
    base = {
        key: value.clone()
        for key, value in checkpoint["state_dict"].items()
    }
    return checkpoint, bundle, model, names, base


def _score(model, state, bundle, device, names, bpw_map):
    from evaluate_quant import _model_size_mb
    from train import evaluate

    model.load_state_dict(state)
    model.to(device)
    metrics = evaluate(model, bundle.test, device, bundle.n_classes)
    size = _model_size_mb(model, names, bpw_map)
    return metrics, size


def run_fixed(checkpoints: list[Path] | None = None) -> None:
    from quantizers import lloyd_max_codebook, quant_fp16, quant_int8_minmax
    from train import get_device

    device = get_device()
    paths = checkpoints or sorted(CKPT_DIR.glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"no checkpoints found in {CKPT_DIR}")
    codebooks = {bits: lloyd_max_codebook(2**bits) for bits in BITS}
    rows = []

    for path in paths:
        checkpoint, bundle, model, names, base = _load_checkpoint(path, device)
        dataset, model_seed = checkpoint["dataset"], checkpoint["seed"]
        protected = {names[0], names[-1]}
        checkpoint_hash = _sha256(path)

        def record(method, bits, state, bpw_map, rate, rotation_seed="-"):
            metrics, size = _score(
                model,
                state,
                bundle,
                device,
                names,
                bpw_map,
            )
            row = {
                "dataset": dataset,
                "model_seed": model_seed,
                "rotation_seed": rotation_seed,
                "method": method,
                "bits": bits,
                "bpw": round(rate, 4),
                "size_mb": round(size, 4),
                "checkpoint_sha256": checkpoint_hash,
                **{key: round(value, 6) for key, value in metrics.items()},
            }
            rows.append(row)
            print(
                f"[{dataset} model={model_seed}] {method}_{bits} "
                f"auc={metrics['auc']:.4f} acc={metrics['acc']:.4f}"
            )

        record("fp32", 32, base, {name: 32.0 for name in names}, 32.0)
        for method, quantizer, bits in (
            ("fp16", quant_fp16, 16),
            ("int8_minmax", quant_int8_minmax, 8),
        ):
            state = {key: value.clone() for key, value in base.items()}
            bpw_map = {}
            for name in names:
                result = quantizer(base[name].cpu().numpy())
                state[name] = __import__("torch").from_numpy(result.weight)
                bpw_map[name] = result.bits_per_weight
            rate = float(
                np.average(
                    list(bpw_map.values()),
                    weights=[base[name].numel() for name in names],
                )
            )
            record(method, bits, state, bpw_map, rate)

        for bits in BITS:
            methods = list(PRIMARY_METHODS)
            if bits == 4:
                methods = list(ABLATION_METHODS)
            for method in methods:
                state, bpw_map, rate = _apply_fixed(
                    base,
                    names,
                    method,
                    bits,
                    protected=protected,
                    codebook=codebooks[bits],
                    rotation_seed=0,
                )
                record(method, bits, state, bpw_map, rate, rotation_seed=0)

    RESULTS_DIR.mkdir(exist_ok=True)
    with FIXED_RESULTS.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {FIXED_RESULTS}")


def run_factorial(
    rotation_seeds=DEFAULT_ROTATION_SEEDS,
    checkpoints: list[Path] | None = None,
) -> None:
    from quantizers import lloyd_max_codebook
    from train import get_device

    device = get_device()
    paths = checkpoints or sorted(CKPT_DIR.glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"no checkpoints found in {CKPT_DIR}")
    codebook = lloyd_max_codebook(16)
    rows = []
    for path in paths:
        checkpoint, bundle, model, names, base = _load_checkpoint(path, device)
        dataset, model_seed = checkpoint["dataset"], checkpoint["seed"]
        protected = {names[0], names[-1]}
        checkpoint_hash = _sha256(path)
        for rotation_seed in rotation_seeds:
            state, bpw_map, rate = _apply_fixed(
                base,
                names,
                "rotated_normal_codebook",
                4,
                protected=protected,
                codebook=codebook,
                rotation_seed=rotation_seed,
            )
            metrics, size = _score(
                model,
                state,
                bundle,
                device,
                names,
                bpw_map,
            )
            rows.append(
                {
                    "dataset": dataset,
                    "model_seed": model_seed,
                    "rotation_seed": rotation_seed,
                    "method": "rotated_normal_codebook",
                    "bits": 4,
                    "bpw": round(rate, 4),
                    "size_mb": round(size, 4),
                    "checkpoint_sha256": checkpoint_hash,
                    **{key: round(value, 6) for key, value in metrics.items()},
                }
            )
            print(
                f"[{dataset} model={model_seed} rotation={rotation_seed}] "
                f"auc={metrics['auc']:.4f}"
            )
    RESULTS_DIR.mkdir(exist_ok=True)
    with FACTORIAL_RESULTS.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {FACTORIAL_RESULTS}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--factorial",
        action="store_true",
        help="run the 4-bit model-seed x rotation-seed experiment",
    )
    parser.add_argument(
        "--rotation-seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_ROTATION_SEEDS),
    )
    parser.add_argument("--checkpoints", nargs="*", type=Path)
    args = parser.parse_args()
    if args.factorial:
        run_factorial(args.rotation_seeds, args.checkpoints)
    else:
        run_fixed(args.checkpoints)


if __name__ == "__main__":
    main()
