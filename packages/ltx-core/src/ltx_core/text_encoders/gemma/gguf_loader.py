"""Gemma 3 GGUF state-dict loader — translates llama.cpp tensor names to HuggingFace format.

Gemma 3 GGUFs produced by llama.cpp / Unsloth use a flat naming scheme
(``blk.N.attn_q.weight``) that differs from HuggingFace safetensors checkpoints
(``language_model.model.layers.N.self_attn.q_proj.weight``).  This module
provides the translation layer so that existing ``GEMMA_LLM_KEY_OPS`` and
``GemmaTextEncoderConfigurator`` continue to work unchanged.

Only language-model tensors are mapped; vision-tower and multimodal-projector
tensors present in chat-tuned GGUFs are silently skipped — they are not
used for text-to-video generation.
"""
from __future__ import annotations

import re

import torch

from ltx_core.loader.primitives import StateDict

try:
    from gguf import GGUFReader
except ImportError as _e:
    raise ImportError("pip install gguf  (>= 0.9.0) required for GGUF Gemma loading") from _e

from ltx_core.loader.gguf_loader import _to_bf16  # reuse dequantization helper

# ---------------------------------------------------------------------------
# Mapping tables
# ---------------------------------------------------------------------------

# Per-block suffix mapping: llama.cpp suffix → HuggingFace suffix (within layer).
# Verified against gemma-3-12b-it-qat-UD-Q4_K_XL.gguf (Unsloth/llama.cpp export).
_BLOCK_SUFFIX_MAP: dict[str, str] = {
    # --- Norms (Gemma 3 has 4 norms per block) ---
    "attn_norm.weight":          "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "ffn_norm.weight":           "pre_feedforward_layernorm.weight",
    "post_ffw_norm.weight":      "post_feedforward_layernorm.weight",
    # --- Attention projections ---
    "attn_q.weight":      "self_attn.q_proj.weight",
    "attn_k.weight":      "self_attn.k_proj.weight",
    "attn_v.weight":      "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    # --- QK-norm (Gemma 3) ---
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    # --- MLP ---
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight":   "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    # --- Biases (rare in Gemma, included for completeness) ---
    "attn_q.bias":      "self_attn.q_proj.bias",
    "attn_k.bias":      "self_attn.k_proj.bias",
    "attn_v.bias":      "self_attn.v_proj.bias",
    "attn_output.bias": "self_attn.o_proj.bias",
    "ffn_gate.bias":    "mlp.gate_proj.bias",
    "ffn_up.bias":      "mlp.up_proj.bias",
    "ffn_down.bias":    "mlp.down_proj.bias",
}

# Non-block (global) tensor mapping
_GLOBAL_MAP: dict[str, str] = {
    "token_embd.weight": "language_model.model.embed_tokens.weight",
    "output_norm.weight": "language_model.model.norm.weight",
    # output.weight is weight-tied to token_embd in Gemma; skip it — the KV
    # operation in GEMMA_LLM_KEY_OPS duplicates embed_tokens → lm_head.
}

_BLK_RE = re.compile(r"^blk\.(\d+)\.(.+)$")


def gemma_gguf_key_to_hf(key: str) -> str | None:
    """Map a llama.cpp Gemma GGUF tensor name to its HuggingFace checkpoint key.

    Returns ``None`` for keys that have no HuggingFace equivalent
    (vision tower, multimodal projector, ``output.weight``, …).  Those
    tensors are silently skipped at load time.
    """
    if key in _GLOBAL_MAP:
        return _GLOBAL_MAP[key]
    m = _BLK_RE.match(key)
    if m:
        n, suffix = m.group(1), m.group(2)
        hf_suffix = _BLOCK_SUFFIX_MAP.get(suffix)
        if hf_suffix:
            return f"language_model.model.layers.{n}.{hf_suffix}"
    return None


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class GemmaGGUFStateDictLoader:
    """StateDictLoader for Gemma 3 GGUF files (llama.cpp / Unsloth quantised format).

    Translates llama.cpp tensor names to HuggingFace ``Gemma3ForConditionalGeneration``
    checkpoint format and then applies ``sd_ops`` (typically ``GEMMA_LLM_KEY_OPS``)
    to produce the final LTX-internal key names.

    ``metadata()`` always returns ``{}`` — ``GemmaTextEncoderConfigurator.from_config``
    ignores the config dict and uses the hardcoded ``GEMMA3_CONFIG_FOR_LTX``.

    Usage::

        from ltx_core.text_encoders.gemma.gguf_loader import GemmaGGUFStateDictLoader
        from ltx_core.text_encoders.gemma import GEMMA_LLM_KEY_OPS

        loader = GemmaGGUFStateDictLoader()
        sd = loader.load("gemma-3-12b-it-qat-UD-Q4_K_XL.gguf", sd_ops=GEMMA_LLM_KEY_OPS)
    """

    def metadata(self, path: str) -> dict:  # noqa: ARG002
        return {}

    def load(
        self,
        path: str | list[str],
        sd_ops=None,
        device: torch.device | None = None,
    ) -> StateDict:
        if isinstance(path, list):
            if len(path) != 1:
                raise ValueError(
                    f"GemmaGGUFStateDictLoader expects a single GGUF file; got {path}"
                )
            path = path[0]

        device = device or torch.device("cpu")
        reader = GGUFReader(path)
        sd: dict[str, torch.Tensor] = {}
        total_bytes = 0
        all_dtypes: set[torch.dtype] = set()

        for tensor in reader.tensors:
            hf_key = gemma_gguf_key_to_hf(tensor.name)
            if hf_key is None:
                continue

            t = _to_bf16(tensor.name, tensor.data, tensor.tensor_type).to(device)

            if sd_ops is not None:
                model_key = sd_ops.apply_to_key(hf_key)
                if model_key is None:
                    continue
                for out_key, out_val in sd_ops.apply_to_key_value(model_key, t):
                    sd[out_key] = out_val
                    total_bytes += out_val.nbytes
                    all_dtypes.add(out_val.dtype)
            else:
                sd[hf_key] = t
                total_bytes += t.nbytes
                all_dtypes.add(t.dtype)

        return StateDict(sd=sd, device=device, size=total_bytes, dtype=all_dtypes)
