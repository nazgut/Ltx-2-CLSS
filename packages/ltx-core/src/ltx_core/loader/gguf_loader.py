"""GGUF state-dict loader for LTX-core.

Dequantizes tensors from a GGUF checkpoint file (Q4_K, Q5_K, Q8_0, F16, BF16, F32 …)
to BF16 at load time, exposing the same ``StateDictLoader`` interface as
``SafetensorsModelStateDictLoader``.  No conversion to disk is required.

Memory note
-----------
The full dequantized model (BF16) is loaded into CPU RAM.  For the LTX-2.3
22B model the Q4_K_S GGUF is ~11 GB on disk but ~44 GB once expanded to BF16.
Make sure the host has ≥ 48 GB of free RAM before loading.  To avoid that cost
the companion script ``convert_gguf.py`` can produce persistent BF16 safetensors
shards (load-once, fast subsequent inference).
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import torch

from ltx_core.loader.primitives import StateDict

if TYPE_CHECKING:
    from ltx_core.loader.sd_ops import SDOps

try:
    from gguf import GGUFReader
    from gguf.constants import GGMLQuantizationType
    from gguf.quants import dequantize as _gguf_dequantize
except ImportError as _e:
    raise ImportError(
        "The 'gguf' package is required for direct GGUF loading.\n"
        "Install it with:  pip install gguf  (version ≥ 0.9.0)\n"
        f"Original error: {_e}"
    ) from _e

_FLOAT_QTYPES = frozenset({
    GGMLQuantizationType.F32,
    GGMLQuantizationType.F16,
    GGMLQuantizationType.BF16,
})


def _to_bf16(name: str, raw: np.ndarray, qtype: GGMLQuantizationType) -> torch.Tensor:
    if qtype == GGMLQuantizationType.F32:
        return torch.from_numpy(raw).to(torch.bfloat16)
    if qtype == GGMLQuantizationType.F16:
        return torch.from_numpy(raw.astype(np.float32)).to(torch.bfloat16)
    if qtype == GGMLQuantizationType.BF16:
        return torch.from_numpy(raw.view(np.uint16)).view(torch.bfloat16)
    try:
        fp32 = _gguf_dequantize(raw, qtype)
        return torch.from_numpy(fp32).to(torch.bfloat16)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to dequantize GGUF tensor '{name}' (type {qtype.name}): {exc}"
        ) from exc


class CombinedStateDictLoader:
    """Merge state dicts from multiple (path, loader) pairs.

    Loads tensors from all provided sources, applying the same ``sd_ops`` to
    each, and merges the results into a single ``StateDict``.  Duplicate keys
    from later sources overwrite earlier ones.

    ``metadata()`` returns the first non-empty config found across all sources
    (tried in order).  This lets the caller put the authoritative source first.

    Usage::

        loader = CombinedStateDictLoader([
            (gguf_path,  GGUFStateDictLoader()),     # connector weights + config
            (sft_path,   SafetensorsStateDictLoader()),  # projection weights
        ])
        builder = SingleGPUModelBuilder(
            model_path=gguf_path,
            model_loader=loader,
            ...
        )

    Note: the ``path`` argument to ``load()`` is ignored; the constructor-
    supplied paths are used instead.
    """

    def __init__(self, loaders_and_paths: list[tuple[str, "StateDictLoader"]]) -> None:
        self._loaders_and_paths = loaders_and_paths

    def metadata(self, path: str) -> dict:  # noqa: ARG002
        for lpath, loader in self._loaders_and_paths:
            meta = loader.metadata(lpath)
            if meta:
                return meta
        return {}

    def load(
        self,
        path: str | list[str],  # ignored — uses constructor paths  # noqa: ARG002
        sd_ops: "SDOps | None" = None,
        device: torch.device | None = None,
    ) -> "StateDict":
        combined: dict[str, torch.Tensor] = {}
        total_bytes = 0
        all_dtypes: set[torch.dtype] = set()
        device = device or torch.device("cpu")
        for lpath, loader in self._loaders_and_paths:
            sd = loader.load(lpath, sd_ops=sd_ops, device=device)
            combined.update(sd.sd)
            total_bytes += sd.size
            all_dtypes |= sd.dtype
        return StateDict(sd=combined, device=device, size=total_bytes, dtype=all_dtypes)


class ConfigOverrideSafetensorsLoader:
    """Safetensors loader that returns an externally-supplied config dict from ``metadata()``.

    Use this when a safetensors file does not embed its own ``config`` metadata
    (e.g. the LTX embeddings-connectors file) but the config is known from another
    source (e.g. the transformer's GGUF header).

    Parameters
    ----------
    config :
        Config dict to return from ``metadata()``.  Must match the format
        expected by the target ``ModelConfigurator.from_config()`` call.
    """

    def __init__(self, config: dict) -> None:
        self._config = config

    def metadata(self, path: str) -> dict:  # noqa: ARG002
        return self._config

    def load(
        self,
        path: str | list[str],
        sd_ops: "SDOps | None" = None,
        device: torch.device | None = None,
    ) -> "StateDict":
        from ltx_core.loader.sft_loader import SafetensorsStateDictLoader
        return SafetensorsStateDictLoader().load(path, sd_ops=sd_ops, device=device)


class GGUFStateDictLoader:
    """Load model weights and config from a GGUF checkpoint.

    Conforms to the ``StateDictLoader`` protocol so it can be passed to
    ``SingleGPUModelBuilder(model_loader=GGUFStateDictLoader())`` or
    ``StreamingModelBuilder(model_loader=GGUFStateDictLoader())``.
    """

    def metadata(self, path: str) -> dict:
        """Extract LTX model config from the GGUF file header.

        The GGUF file is expected to have a ``"config"`` metadata field
        containing a JSON-encoded model configuration as a byte array (the
        same format written by ``convert_gguf.py``).
        """
        reader = GGUFReader(path)
        config_field = reader.fields.get("config")
        if config_field is None:
            return {}
        parts = config_field.parts[-1]
        if hasattr(parts, "tolist"):
            raw = parts.tolist()
            if isinstance(raw, list):
                try:
                    return json.loads(bytes(raw).decode("utf-8"))
                except Exception:
                    pass
        # Fallback: try string field
        try:
            if parts.dtype.kind in ("U", "S"):
                return json.loads(str(parts[0]))
        except Exception:
            pass
        return {}

    def load(
        self,
        path: str | list[str],
        sd_ops: "SDOps | None" = None,
        device: torch.device | None = None,
    ) -> StateDict:
        """Dequantize all tensors from the GGUF file and return a ``StateDict``.

        ``path`` may be a string or a single-element list (GGUF files are never
        sharded).  Multi-element lists raise ``ValueError``.
        """
        if isinstance(path, list):
            if len(path) != 1:
                raise ValueError(
                    f"GGUFStateDictLoader expects a single file path; got {path}"
                )
            path = path[0]

        device = device or torch.device("cpu")
        reader = GGUFReader(path)

        sd: dict[str, torch.Tensor] = {}
        total_bytes = 0
        all_dtypes: set[torch.dtype] = set()

        for tensor in reader.tensors:
            gguf_key = tensor.name
            t = _to_bf16(gguf_key, tensor.data, tensor.tensor_type).to(device)

            if sd_ops is not None:
                model_key = sd_ops.apply_to_key(gguf_key)
                if model_key is None:
                    continue
                for out_key, out_val in sd_ops.apply_to_key_value(model_key, t):
                    sd[out_key] = out_val
                    total_bytes += out_val.nbytes
                    all_dtypes.add(out_val.dtype)
            else:
                sd[gguf_key] = t
                total_bytes += t.nbytes
                all_dtypes.add(t.dtype)

        return StateDict(sd=sd, device=device, size=total_bytes, dtype=all_dtypes)
