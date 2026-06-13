"""GPU-accelerated GGUF dequantization.

Replaces the CPU numpy path (_to_bf16) for the hot per-block dequantization
path in GGUFBlockReader. Raw quantized bytes are moved to GPU as compact uint8
tensors, then dequantized using PyTorch tensor operations.

Two interfaces:
- dequant_gpu_tensor(raw_gpu, qtype, out_shape): input is already a GPU uint8
  slice — used by GGUFDirectGPUReader after a single batched H2D transfer.
- dequant_gpu(raw_np, qtype, device, out_shape): input is a CPU numpy array —
  used for occasional non-block tensors (loaded once, not on the hot path).

Algorithms are direct ports of gguf/quants.py dequantize_blocks methods.
Block sizes and field layouts verified against that source.
"""
from __future__ import annotations

import numpy as np
import torch

try:
    from gguf.constants import GGMLQuantizationType
    _GGUF_AVAILABLE = True
except ImportError:
    GGMLQuantizationType = None  # type: ignore[assignment]
    _GGUF_AVAILABLE = False

# GGUF super-block element count for K-quants
_QK_K = 256


# ---------------------------------------------------------------------------
# Q8_0  (block_size=34, QK=32)
# ---------------------------------------------------------------------------

def _dequant_q8_0(raw: torch.Tensor, out_shape: tuple) -> torch.Tensor:
    """Q8_0: 34-byte blocks. Layout: [d:2 f16][x:32 i8]. 32 elements/block."""
    BLOCK = 34
    n = raw.numel() // BLOCK
    b = raw.reshape(n, BLOCK)

    d = b[:, :2].contiguous().view(torch.float16).float()      # (n, 1)
    x = b[:, 2:].contiguous().view(torch.int8).float()         # (n, 32)

    return (x * d).reshape(out_shape).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Shared scale unpacker for Q4_K and Q5_K
# ---------------------------------------------------------------------------

def _q4k_get_scale_min(scales_u8: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack 12-byte Q4_K scales → 8 (sc, min) pairs per super-block (int32).

    Layout (verified against gguf/quants.py Q4_K.get_scale_min):
        reshape to (n, 3, 4) → d_bytes, m_bytes, md_bytes
        sc  = [d & 0x3F, (md & 0x0F) | ((d >> 2) & 0x30)]   8 values
        min = [m & 0x3F, (md >> 4)   | ((m >> 2) & 0x30)]   8 values
    """
    n = scales_u8.shape[0]
    s = scales_u8.to(torch.int32).reshape(n, 3, 4)
    d_b  = s[:, 0, :]   # (n, 4)
    m_b  = s[:, 1, :]   # (n, 4)
    md_b = s[:, 2, :]   # (n, 4)

    sc  = torch.cat([d_b & 0x3F,  (md_b & 0x0F) | ((d_b  >> 2) & 0x30)], dim=1)   # (n, 8)
    mn  = torch.cat([m_b & 0x3F,  (md_b >> 4)   | ((m_b  >> 2) & 0x30)], dim=1)   # (n, 8)
    return sc, mn


# ---------------------------------------------------------------------------
# Q4_K  (block_size=144, QK_K=256)
# ---------------------------------------------------------------------------

def _dequant_q4_k(raw: torch.Tensor, out_shape: tuple) -> torch.Tensor:
    """Q4_K: 144-byte super-blocks, 256 elements.

    Layout: [d:2 f16][dmin:2 f16][scales:12][qs:128]
    """
    BLOCK = 144
    n = raw.numel() // BLOCK
    b = raw.reshape(n, BLOCK)

    d    = b[:, :2].contiguous().view(torch.float16).float()       # (n, 1)
    dmin = b[:, 2:4].contiguous().view(torch.float16).float()      # (n, 1)
    sc, mn = _q4k_get_scale_min(b[:, 4:16].contiguous())           # (n, 8) int32 each

    d_g  = (d * sc.float()).reshape(n, 8, 1)                       # (n, 8, 1)
    dm_g = (dmin * mn.float()).reshape(n, 8, 1)                    # (n, 8, 1)

    # Unpack 4-bit quants: 128 bytes → 256 values → (n, 8, 32)
    qs = b[:, 16:].contiguous().to(torch.int32).reshape(n, 4, 32)
    q = torch.stack([qs & 0x0F, (qs >> 4) & 0x0F], dim=2).reshape(n, 8, 32).float()

    return (d_g * q - dm_g).reshape(out_shape).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Q5_K  (block_size=176, QK_K=256)
# ---------------------------------------------------------------------------

def _dequant_q5_k(raw: torch.Tensor, out_shape: tuple) -> torch.Tensor:
    """Q5_K: 176-byte super-blocks, 256 elements.

    Layout: [d:2 f16][dmin:2 f16][scales:12][qh:32][qs:128]
    """
    BLOCK = 176
    n = raw.numel() // BLOCK
    b = raw.reshape(n, BLOCK)

    d    = b[:, :2].contiguous().view(torch.float16).float()       # (n, 1)
    dmin = b[:, 2:4].contiguous().view(torch.float16).float()      # (n, 1)
    sc, mn = _q4k_get_scale_min(b[:, 4:16].contiguous())           # (n, 8) int32

    d_g  = (d * sc.float()).reshape(n, 8, 1)                       # (n, 8, 1)
    dm_g = (dmin * mn.float()).reshape(n, 8, 1)                    # (n, 8, 1)

    # Low 4 bits from qs: (n, 128) → (n, 8, 32)
    qs_i = b[:, 48:].contiguous().to(torch.int32).reshape(n, 4, 32)
    ql = torch.stack([qs_i & 0x0F, (qs_i >> 4) & 0x0F], dim=2).reshape(n, 8, 32)

    # High bit from qh: (n, 32) → unpack 8 bits per byte → (n, 8, 32)
    qh_i = b[:, 16:48].contiguous().to(torch.int32).reshape(n, 1, 32)
    shifts = torch.arange(8, device=raw.device, dtype=torch.int32).reshape(1, 8, 1)
    qh = (qh_i >> shifts) & 0x01                                   # (n, 8, 32)

    q = (ql | (qh << 4)).float()                                   # 5-bit values 0..31

    return (d_g * q - dm_g).reshape(out_shape).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Q6_K  (block_size=210, QK_K=256)
# ---------------------------------------------------------------------------

def _dequant_q6_k(raw: torch.Tensor, out_shape: tuple) -> torch.Tensor:
    """Q6_K: 210-byte super-blocks, 256 elements.

    Layout: [ql:128][qh:64][scales:16 i8][d:2 f16]
    """
    BLOCK = 210
    n = raw.numel() // BLOCK
    b = raw.reshape(n, BLOCK)

    # 4 low bits: 128 bytes → (n, 2, 64) → unpack lo/hi → (n, 8, 32)
    ql_i = b[:, :128].contiguous().to(torch.int32).reshape(n, 2, 64)
    ql = torch.stack([ql_i & 0x0F, (ql_i >> 4) & 0x0F], dim=2).reshape(n, 8, 32)

    # 2 high bits: 64 bytes → (n, 2, 1, 32), unpack [0,2,4,6] → (n, 2, 4, 32) → (n, 8, 32)
    qh_i = b[:, 128:192].contiguous().to(torch.int32).reshape(n, 2, 1, 32)
    shifts_qh = torch.tensor([0, 2, 4, 6], device=raw.device, dtype=torch.int32).reshape(1, 1, 4, 1)
    qh = ((qh_i >> shifts_qh) & 0x03).reshape(n, 8, 32)

    # 6-bit values: combine → subtract 32 → signed int
    q = (ql | (qh << 4)) - 32                                      # (n, 8, 32) int32 -32..31

    # int8 scales: 16 bytes → (n, 16), d: 2 bytes → (n, 1)
    sc = b[:, 192:208].contiguous().view(torch.int8).to(torch.float32)  # (n, 16)
    d  = b[:, 208:210].contiguous().view(torch.float16).float()         # (n, 1)

    d_g = (d * sc).reshape(n, 16, 1)                               # (n, 16, 1)
    q_f = q.float().reshape(n, 16, 16)                             # (n, 16, 16)

    return (d_g * q_f).reshape(out_shape).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Dispatch — GPU uint8 tensor input (primary hot-path interface)
# ---------------------------------------------------------------------------

_GPU_DISPATCH = None  # populated lazily after GGMLQuantizationType is available


def _build_dispatch():
    global _GPU_DISPATCH
    if not _GGUF_AVAILABLE:
        _GPU_DISPATCH = {}
        return
    _GPU_DISPATCH = {
        GGMLQuantizationType.Q4_K: _dequant_q4_k,
        GGMLQuantizationType.Q5_K: _dequant_q5_k,
        GGMLQuantizationType.Q6_K: _dequant_q6_k,
        GGMLQuantizationType.Q8_0: _dequant_q8_0,
    }


def dequant_gpu_tensor(
    raw: torch.Tensor,
    qtype,
    out_shape: tuple,
) -> torch.Tensor | None:
    """Dequantize a GPU uint8 slice on-device.

    *raw* must already be on the target GPU (a flat uint8 tensor produced by
    slicing the block's batched H2D buffer).  Returns ``None`` for types not
    handled here so the caller can fall back.
    """
    global _GPU_DISPATCH
    if _GPU_DISPATCH is None:
        _build_dispatch()
    fn = _GPU_DISPATCH.get(qtype)  # type: ignore[arg-type]
    return fn(raw, out_shape) if fn is not None else None


# ---------------------------------------------------------------------------
# Legacy CPU-numpy interface (used for non-block/one-off tensors)
# ---------------------------------------------------------------------------

def dequant_gpu(
    raw: np.ndarray,
    qtype,
    device: torch.device,
    out_shape: tuple,
) -> torch.Tensor | None:
    """Dequantize a GGUF tensor on GPU from a CPU numpy source.

    Copies raw bytes to GPU then dequantizes.  Returns ``None`` for unsupported
    types so the caller can fall back to :func:`~ltx_core.loader.gguf_loader._to_bf16`.
    """
    global _GPU_DISPATCH
    if _GPU_DISPATCH is None:
        _build_dispatch()
    if not _GPU_DISPATCH:
        return None
    fn = _GPU_DISPATCH.get(qtype)  # type: ignore[arg-type]
    if fn is None:
        return None
    flat_np = np.frombuffer(raw, dtype=np.uint8)
    gpu = torch.from_numpy(flat_np.copy()).to(device, non_blocking=True)
    return fn(gpu, out_shape)
