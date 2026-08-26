"""Write a reproducibility manifest (results/dfq_corrected_manifest.json) recording the
environment that produced the result CSVs: package versions, device, platform,
seeds, and a content hash + mtime of each result file. Run this right after the
evaluation grid so the manifest sits next to the CSVs it describes.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

RES = Path(__file__).parent.parent / "results"
MODEL_SEEDS = [42, 123, 456]
ROTATION_SEEDS = [0, 1, 2, 3, 4]
RESULT_FILES = [
    "dfq_corrected_fixed.csv",
    "dfq_rotation_factorial.csv",
    "dfq_corrected_summary.md",
]
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"


def _version(mod: str) -> str | None:
    try:
        return __import__(mod).__version__
    except Exception:
        return None


def _device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.get_device_name(0)}"
        return "cpu"
    except Exception:
        return "unknown"


def _sha256(p: Path) -> str | None:
    if not p.exists():
        return None
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def main() -> None:
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "device": _device(),
        "protocol": "fixed matched-cardinality 2x2; no low-bit validation selection",
        "model_seeds": MODEL_SEEDS,
        "rotation_seeds": ROTATION_SEEDS,
        "packages": {
            m: _version(m)
            for m in ["numpy", "pandas", "torch", "torchvision",
                      "medmnist", "matplotlib", "sklearn"]
        },
        "result_files": {
            f: {"sha256": _sha256(RES / f),
                "mtime_utc": (datetime.fromtimestamp((RES / f).stat().st_mtime,
                                                     timezone.utc).isoformat()
                              if (RES / f).exists() else None)}
            for f in RESULT_FILES
        },
        "checkpoints": {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in sorted(CHECKPOINT_DIR.glob("*.pt"))
        },
    }
    out = RES / "dfq_corrected_manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
