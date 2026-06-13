"""
convert_gguf.py  —  Convert a GGUF checkpoint to sharded BF16 safetensors
                     so it can be loaded by the LTX-2 pipeline.

Usage
-----
    python convert_gguf.py \
        --input  ltx-2.3-22b-dev-UD-Q4_K_S.gguf \
        --output ./ltx-2.3-22b-bf16/          \
        [--shard-size-gb 4]

Requires
--------
    pip install gguf safetensors tqdm

Notes
-----
- A Q4_K_S GGUF of the 22B model is ~11 GB.  Dequantized BF16 output is ~44 GB.
  Make sure you have at least 60 GB free disk space and ~50 GB free RAM during
  conversion (numpy intermediate buffers).
- The script converts shard-by-shard to keep peak RAM reasonable (~8 GB per
  shard at the default 4 GB shard size).
- Tensors that are already F32/F16/BF16 in the GGUF are converted directly.
- Output directory will contain:
      model.safetensors.index.json   (weight map)
      model-00001-of-XXXXX.safetensors
      model-00002-of-XXXXX.safetensors
      ...
      config.json                    (metadata extracted from GGUF)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from tqdm import tqdm

try:
    from gguf import GGUFReader
    from gguf.constants import GGMLQuantizationType
    from gguf.quants import dequantize
except ImportError:
    sys.exit(
        "ERROR: 'gguf' package not found.\n"
        "Install it with:  pip install gguf\n"
        "(version >= 0.9.0 required for Q4_K dequantization)"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FLOAT_TYPES = {
    GGMLQuantizationType.F32,
    GGMLQuantizationType.F16,
    GGMLQuantizationType.BF16,
}


def _to_bf16(tensor_name: str, raw: np.ndarray, qtype: GGMLQuantizationType) -> torch.Tensor:
    """Return a BF16 torch tensor from GGUF raw data."""
    if qtype in _FLOAT_TYPES:
        if qtype == GGMLQuantizationType.F32:
            return torch.from_numpy(raw).to(torch.bfloat16)
        if qtype == GGMLQuantizationType.F16:
            return torch.from_numpy(raw.astype(np.float32)).to(torch.bfloat16)
        if qtype == GGMLQuantizationType.BF16:
            return torch.from_numpy(raw.view(np.uint16)).view(torch.bfloat16)
    # Quantised types: use gguf's dequantize() → float32 → bfloat16
    try:
        fp32 = dequantize(raw, qtype)          # float32 numpy array
        return torch.from_numpy(fp32).to(torch.bfloat16)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to dequantize tensor '{tensor_name}' "
            f"(type {qtype.name}): {exc}"
        ) from exc


def _bytes_of(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


# ---------------------------------------------------------------------------
# Main conversion logic
# ---------------------------------------------------------------------------

def convert(input_path: str, output_dir: str, shard_size_gb: float = 4.0) -> None:
    shard_bytes = int(shard_size_gb * 1024**3)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Reading GGUF file: {input_path}")
    reader = GGUFReader(input_path)

    # Save metadata as config.json
    metadata: dict = {}
    for field in reader.fields.values():
        try:
            parts = field.parts[-1]
            if parts.dtype.kind in ("U", "S"):
                val = str(parts[0])
            elif len(parts) == 1:
                val = parts[0].item()
            else:
                val = parts.tolist()
            metadata[field.name] = val
        except Exception:
            pass
    (out / "config.json").write_text(json.dumps(metadata, indent=2))
    print(f"  Saved metadata → {out / 'config.json'}")
    print(f"  Model: {metadata.get('general.name', 'unknown')}")
    print(f"  Architecture: {metadata.get('general.architecture', 'unknown')}")

    tensors = list(reader.tensors)
    print(f"  Total tensors: {len(tensors)}")

    # -----------------------------------------------------------------------
    # Pass 1: plan shards (estimate output size per tensor)
    # -----------------------------------------------------------------------
    plan: list[tuple[str, GGMLQuantizationType, np.ndarray]] = []
    total_bytes = 0
    for t in tensors:
        # GGUF stores row-major; shapes are stored innermost-first → reverse
        plan.append((t.name, t.tensor_type, t.data))
        # Estimate BF16 output size (2 bytes per element)
        numel = math.prod(t.shape) if t.shape else 1
        total_bytes += numel * 2

    total_gb = total_bytes / 1024**3
    n_shards = max(1, math.ceil(total_bytes / shard_bytes))
    print(f"  Estimated output size: {total_gb:.1f} GB across {n_shards} shards")
    print()

    # -----------------------------------------------------------------------
    # Pass 2: convert and write shards
    # -----------------------------------------------------------------------
    weight_map: dict[str, str] = {}
    shard_idx = 1
    current_shard: dict[str, torch.Tensor] = {}
    current_shard_bytes = 0

    def _flush_shard() -> None:
        nonlocal shard_idx, current_shard, current_shard_bytes
        if not current_shard:
            return
        shard_name = f"model-{shard_idx:05d}-of-{n_shards:05d}.safetensors"
        shard_path = out / shard_name
        print(f"  Writing shard {shard_idx}/{n_shards}: {shard_name} "
              f"({current_shard_bytes / 1024**3:.2f} GB, {len(current_shard)} tensors)")
        save_file(current_shard, str(shard_path))
        for key in current_shard:
            weight_map[key] = shard_name
        shard_idx += 1
        current_shard = {}
        current_shard_bytes = 0

    print("Converting tensors...")
    for name, qtype, raw_data in tqdm(plan, unit="tensor"):
        tensor = _to_bf16(name, raw_data, qtype)
        nbytes = _bytes_of(tensor)

        # Start a new shard if this one would overflow (unless it's the first tensor)
        if current_shard and current_shard_bytes + nbytes > shard_bytes:
            _flush_shard()

        current_shard[name] = tensor
        current_shard_bytes += nbytes

    _flush_shard()  # flush last shard

    # -----------------------------------------------------------------------
    # Write index file
    # -----------------------------------------------------------------------
    index = {
        "metadata": {"total_size": total_bytes},
        "weight_map": weight_map,
    }
    index_path = out / "model.safetensors.index.json"
    index_path.write_text(json.dumps(index, indent=2))
    print(f"\nDone!  Index written to {index_path}")
    print(f"Use --checkpoint-path {out!s} in the generation script.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert GGUF to sharded BF16 safetensors.")
    parser.add_argument("--input", required=True, help="Path to .gguf file.")
    parser.add_argument("--output", required=True, help="Output directory for safetensors shards.")
    parser.add_argument(
        "--shard-size-gb",
        type=float,
        default=4.0,
        help="Maximum size per output shard in GB (default: 4.0).",
    )
    args = parser.parse_args()
    convert(args.input, args.output, args.shard_size_gb)
