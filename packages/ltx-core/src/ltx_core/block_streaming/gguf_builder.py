"""GGUFStreamingModelBuilder — on-demand GGUF block streaming.

Streams transformer blocks from a GGUF file to the GPU one at a time.
Blocks are dequantized from the GGUF file as needed rather than loading
the entire model into RAM upfront, keeping peak CPU memory to
``cpu_slots_count * block_size`` (≈ 2 GB for the LTX 22B model).
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass, field, replace

import numpy as np

import torch
from torch import nn

from ltx_core.block_streaming.builder import _DEFAULT_GPU_SLOTS
from ltx_core.block_streaming.gpu_dequant import dequant_gpu, dequant_gpu_tensor
from ltx_core.block_streaming.pool import WeightPool
from ltx_core.block_streaming.provider import WeightsProvider
from ltx_core.block_streaming.utils import resolve_attr
from ltx_core.block_streaming.wrapper import BlockStreamingWrapper
from ltx_core.loader.fuse_loras import FuseRule, bf16_fuse_rule
from ltx_core.loader.gguf_loader import GGUFStateDictLoader, _to_bf16
from ltx_core.loader.helpers import create_meta_model, read_model_config
from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.primitives import (
    LoraPathStrengthAndSDOps,
    ModelBuilderProtocol,
    TensorLayout,
)
from ltx_core.loader.registry import DummyRegistry, Registry
from ltx_core.loader.sd_ops import SDOps
from ltx_core.model.model_protocol import ModelConfigurator, ModelType

_DEFAULT_CPU_SLOTS = 2

logger = logging.getLogger(__name__)


def _scan_gguf_keys(
    path: str,
    sd_ops: SDOps | None,
    blocks_prefix: str,
    key_mapper=None,
) -> tuple[dict[int, list[tuple[str, str]]], list[tuple[str, str]]]:
    """Partition GGUF tensor names into per-block and non-block lists.

    Mirrors ``_scan_checkpoint_keys`` in ``builder.py`` but reads tensor names
    from the GGUF file header instead of a safetensors file.

    Parameters
    ----------
    key_mapper :
        Optional ``(gguf_key) -> str | None`` callable applied *before*
        ``sd_ops``.  Use this to translate llama.cpp key names (e.g. Gemma)
        to HuggingFace format so that existing ``SDOps`` objects work unchanged.
        Return ``None`` to skip the tensor entirely.

    Returns
    -------
    block_key_map : {block_idx: [(gguf_key, param_name), ...]}
    non_block_keys : [(gguf_key, model_key), ...]
    """
    try:
        from gguf import GGUFReader
    except ImportError as exc:
        raise ImportError("pip install gguf  (≥ 0.9.0) required") from exc

    reader = GGUFReader(path)
    block_key_map: dict[int, list[tuple[str, str]]] = {}
    non_block_keys: list[tuple[str, str]] = []
    prefix_dot = blocks_prefix + "."

    for tensor in reader.tensors:
        gguf_key = tensor.name
        if key_mapper is not None:
            mapped_key = key_mapper(gguf_key)
            if mapped_key is None:
                continue
        else:
            mapped_key = gguf_key
        model_key = sd_ops.apply_to_key(mapped_key) if sd_ops is not None else mapped_key
        if model_key is None:
            continue
        if model_key.startswith(prefix_dot):
            rest = model_key[len(prefix_dot):]
            idx_str, _, param_name = rest.partition(".")
            try:
                block_idx = int(idx_str)
            except ValueError:
                non_block_keys.append((gguf_key, model_key))
                continue
            block_key_map.setdefault(block_idx, []).append((gguf_key, param_name))
        else:
            non_block_keys.append((gguf_key, model_key))

    return block_key_map, non_block_keys


class GGUFDirectGPUReader:
    """Dequantizes GGUF block tensors directly into GPU target buffers.

    All raw bytes for an entire block are gathered into a single pinned CPU
    buffer and transferred to GPU in one batched H2D copy, then dequantized
    on-device using PyTorch tensor operations.  This avoids both per-tensor H2D
    overhead (~86 calls per block → 1) and the CPU numpy dequantization (~2.5 s
    → ~50 ms per block).

    Falls back to :func:`_to_bf16` (CPU) for BF16 / F16 / F32 tensors that
    appear in blocks (rare, typically <1 % of block data); those are handled
    separately so they don't pollute the batched path.
    """

    def __init__(
        self,
        path: str,
        block_key_map: dict[int, list[tuple[str, str]]],
        device: torch.device,
    ) -> None:
        try:
            from gguf import GGUFReader
        except ImportError as exc:
            raise ImportError("pip install gguf  (>= 0.9.0) required") from exc
        reader = GGUFReader(path)
        self._tensors = {t.name: t for t in reader.tensors}
        self._block_key_map = block_key_map
        self._device = device
        # Pre-allocate a pinned CPU staging buffer large enough for the largest block.
        # We also keep a numpy view of it so we can fill it with a single memcpy
        # (numpy array slice assignment) rather than two copies (read-only → writable → pinned).
        _trivial_types: tuple = ()
        try:
            from gguf.constants import GGMLQuantizationType as _Q
            _trivial_types = (_Q.F32, _Q.F16, _Q.BF16)
        except ImportError:
            pass
        max_block_bytes = max(
            sum(
                int(np.frombuffer(self._tensors[k].data, dtype=np.uint8).shape[0])
                for k, _p in entries
                if k in self._tensors
                and self._tensors[k].tensor_type not in _trivial_types
            )
            for entries in block_key_map.values()
        )
        self._staging = torch.empty(max_block_bytes, dtype=torch.uint8, pin_memory=True)
        self._staging_np = self._staging.numpy()  # zero-copy numpy view for fast fills

    def read_into(self, target: dict[str, torch.Tensor], block_idx: int) -> None:
        """Dequantize block *block_idx* directly into the GPU *target* tensors.

        Strategy:
        1. Copy raw quantized bytes for all block tensors into *one* pinned CPU
           staging buffer.
        2. One non-blocking H2D transfer of the whole block (pinned → GPU).
        3. Dequantize each tensor from its GPU slice using :mod:`gpu_dequant`.
        """
        entries = self._block_key_map[block_idx]

        try:
            from gguf.constants import GGMLQuantizationType as _Q
            _trivial = (_Q.F32, _Q.F16, _Q.BF16)
        except ImportError:
            _trivial = ()

        # --- Phase 1: fill pinned staging buffer (CPU-side, one memcpy per tensor) ---
        offsets: list[tuple[str, str, object, int, int]] = []  # (gguf_key, param, qtype, start, size)
        cpu_fallback: list[tuple[str, torch.Tensor]] = []
        write_pos = 0
        for gguf_key, param_name in entries:
            tensor = self._tensors.get(gguf_key)
            if tensor is None or param_name not in target:
                continue
            if tensor.tensor_type in _trivial:
                t_cpu = _to_bf16(tensor.name, tensor.data, tensor.tensor_type)
                cpu_fallback.append((param_name, t_cpu))
                continue
            raw_np = np.frombuffer(tensor.data, dtype=np.uint8)
            n_bytes = raw_np.shape[0]
            # Direct numpy assignment into pinned buffer (one copy, no extra allocation)
            self._staging_np[write_pos:write_pos + n_bytes] = raw_np
            offsets.append((gguf_key, param_name, tensor.tensor_type, write_pos, n_bytes))
            write_pos += n_bytes

        # --- Phase 2: one batched H2D transfer (pinned → GPU) ---
        if write_pos > 0:
            gpu_buf = self._staging[:write_pos].to(self._device, non_blocking=True)

            # --- Phase 3: dequantize each tensor from its GPU slice ---
            for _gguf_key, param_name, qtype, start, size in offsets:
                raw_gpu = gpu_buf[start:start + size]
                out_shape = target[param_name].shape
                t = dequant_gpu_tensor(raw_gpu, qtype, out_shape)
                if t is None:
                    t = _to_bf16(_gguf_key, self._tensors[_gguf_key].data,
                                  self._tensors[_gguf_key].tensor_type)
                    t = t.to(device=self._device, dtype=torch.bfloat16)
                target[param_name].copy_(t)

        # Apply CPU-fallback tensors (F32 / F16 / BF16, rare in blocks)
        for param_name, t_cpu in cpu_fallback:
            target[param_name].copy_(t_cpu.to(device=self._device, dtype=torch.bfloat16))

    def cleanup(self) -> None:
        self._tensors.clear()


class GGUFGPUWeightSource:
    """Weight source that dequantizes GGUF blocks directly into a GPU pool.

    Replaces the ``DiskWeightSource`` + pinned-CPU-pool pattern.  Blocks are
    dequantized on the GPU (via :class:`GGUFDirectGPUReader`) and cached in a
    small GPU buffer pool.  ``WeightsProvider`` then does a fast GPU→GPU copy
    to the compute pool instead of the usual H2D transfer.

    Dequantization runs on *copy_stream* so that it is naturally sequenced
    before the subsequent GPU→GPU copy that ``WeightsProvider._copy_to_gpu``
    performs on the same stream — no extra sync needed.
    """

    def __init__(
        self,
        pool: WeightPool,
        reader: GGUFDirectGPUReader,
        copy_stream: torch.cuda.Stream,
    ) -> None:
        self._pool = pool
        self._cache: OrderedDict[int, dict[str, torch.Tensor]] = OrderedDict()
        self._events: dict[int, torch.cuda.Event] = {}
        self._reader = reader
        self._copy_stream = copy_stream

    @property
    def block_layout(self) -> TensorLayout:
        return self._pool.buffer_layout

    def get(self, idx: int) -> dict[str, torch.Tensor]:
        """Return GPU weights for block *idx*, dequantizing on cache miss."""
        if idx in self._cache:
            return self._cache[idx]

        if len(self._cache) >= self._pool.capacity:
            evicted_idx, evicted = self._cache.popitem(last=False)
            self._pool.release(evicted, event=self._events.pop(evicted_idx, None))

        weights = self._pool.acquire()
        with torch.cuda.stream(self._copy_stream):
            self._reader.read_into(weights, idx)
        self._cache[idx] = weights
        return weights

    def release(self, idx: int, event: torch.cuda.Event) -> None:
        """Attach a copy-done event; waited before this GPU slot is reused."""
        self._events[idx] = event

    def cleanup(self) -> None:
        self._cache.clear()
        self._events.clear()
        self._reader.cleanup()


@dataclass(frozen=True)
class GGUFStreamingModelBuilder(ModelBuilderProtocol[ModelType]):
    """On-demand GGUF block streaming builder.

    Conforms to ``ModelBuilderProtocol`` and can be passed wherever
    ``StreamingModelBuilder`` is accepted (duck-typed).

    Blocks are dequantized from the GGUF file on demand into a small pinned
    CPU buffer pool (``cpu_slots_count`` slots, default 2).  Only two blocks
    occupy GPU memory at a time via :class:`BlockStreamingWrapper`.

    Parameters
    ----------
    model_class_configurator :
        Used to reconstruct the model architecture from the GGUF config.
    model_path :
        Path to the ``.gguf`` file.  Must be a single string (GGUF is never
        sharded).
    model_sd_ops :
        Key renaming ops applied while loading (e.g.
        ``LTXV_MODEL_COMFY_RENAMING_MAP``).
    module_ops :
        Module-level mutations applied to the meta model.
    loras :
        LoRA adapters fused at build time.
    registry :
        State-dict cache.
    fuse_rule :
        LoRA merge rule; default is ``bf16_fuse_rule``.
    blocks_attr :
        Dotted attribute path to the transformer block list
        (e.g. ``"transformer_blocks"``).
    blocks_prefix :
        State-dict key prefix for block weights
        (e.g. ``"transformer_blocks"``).
    """

    model_class_configurator: type[ModelConfigurator[ModelType]]
    model_path: str
    model_sd_ops: SDOps | None = None
    module_ops: tuple[ModuleOps, ...] = field(default_factory=tuple)
    loras: tuple[LoraPathStrengthAndSDOps, ...] = field(default_factory=tuple)
    registry: Registry = field(default_factory=DummyRegistry)
    fuse_rule: FuseRule = bf16_fuse_rule
    blocks_attr: str = ""
    blocks_prefix: str = ""

    # Internal: always use GGUFStateDictLoader
    @property
    def _loader(self) -> GGUFStateDictLoader:
        return GGUFStateDictLoader()

    @property
    def model_loader(self) -> GGUFStateDictLoader:
        return GGUFStateDictLoader()

    def with_sd_ops(self, sd_ops: SDOps | None) -> "GGUFStreamingModelBuilder":
        return replace(self, model_sd_ops=sd_ops)

    def with_module_ops(self, module_ops: tuple[ModuleOps, ...]) -> "GGUFStreamingModelBuilder":
        return replace(self, module_ops=module_ops)

    def with_loras(self, loras: tuple[LoraPathStrengthAndSDOps, ...]) -> "GGUFStreamingModelBuilder":
        return replace(self, loras=loras)

    def with_registry(self, registry: Registry) -> "GGUFStreamingModelBuilder":
        return replace(self, registry=registry)

    def with_fuse_rule(self, fuse_rule: FuseRule) -> "GGUFStreamingModelBuilder":
        return replace(self, fuse_rule=fuse_rule)

    def with_lora_load_device(self, device: torch.device) -> "GGUFStreamingModelBuilder":
        return self  # no-op; LoRAs are always CPU-loaded

    def model_config(self) -> dict:
        return read_model_config(self.model_path, self._loader)

    def meta_model(
        self,
        config: dict,
        module_ops: tuple[ModuleOps, ...],
    ) -> ModelType:
        return create_meta_model(self.model_class_configurator, config, module_ops)

    def _scan_keys(
        self,
        path: str,
        sd_ops: SDOps | None,
        blocks_prefix: str,
    ) -> tuple[dict[int, list[tuple[str, str]]], list[tuple[str, str]]]:
        """Partition GGUF tensor names into block and non-block groups.
        Subclasses override this to apply a key_mapper before sd_ops.
        """
        return _scan_gguf_keys(path, sd_ops, blocks_prefix)

    def build(
        self,
        target_device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        cpu_slots_count: int | None = None,
        gpu_slots_count: int | None = None,
        **_kwargs: object,
    ) -> BlockStreamingWrapper:
        """Build a :class:`BlockStreamingWrapper` that streams GGUF blocks to GPU on demand.

        Blocks are dequantized directly on the GPU via PyTorch tensor operations
        (no numpy / CPU dequantization in the hot path).  Only the raw quantized
        bytes (~10–90 MB per tensor) are transferred host→device; the BF16 result
        stays on GPU.  Peak CPU RAM is ~14 GB (the raw GGUF file mmap'd by
        GGUFReader) rather than the dequantized model size.
        """
        if not self.blocks_prefix:
            raise ValueError("blocks_prefix must be non-empty for streaming")

        if target_device is None:
            target_device = torch.device("cuda")
        if dtype is None:
            dtype = torch.bfloat16
        if cpu_slots_count is None:
            cpu_slots_count = _DEFAULT_CPU_SLOTS
        if gpu_slots_count is None:
            gpu_slots_count = _DEFAULT_GPU_SLOTS

        loader = self._loader
        config = read_model_config(self.model_path, loader)
        meta_model: nn.Module = create_meta_model(
            self.model_class_configurator, config, self.module_ops
        )
        meta_model.eval()

        blocks = resolve_attr(meta_model, self.blocks_attr)

        block_key_map, non_block_keys = self._scan_keys(
            self.model_path, self.model_sd_ops, self.blocks_prefix
        )

        # --- Derive block layout from meta-model parameter shapes ---
        first_block_idx = min(block_key_map)
        first_block_params = dict(blocks[first_block_idx].named_parameters())
        block_layout: TensorLayout = {
            param_name: (first_block_params[param_name].shape, dtype)
            for _gguf_key, param_name in block_key_map[first_block_idx]
            if param_name in first_block_params
        }

        # --- Load non-block weights directly (small fraction of total size) ---
        try:
            from gguf import GGUFReader
        except ImportError as exc:
            raise ImportError("pip install gguf  (>= 0.9.0) required") from exc
        _reader = GGUFReader(self.model_path)
        _tensor_map = {t.name: t for t in _reader.tensors}
        non_block_sd: dict[str, torch.Tensor] = {}
        for gguf_key, model_key in non_block_keys:
            t = _tensor_map.get(gguf_key)
            if t is None:
                continue
            t_bf16 = _to_bf16(t.name, t.data, t.tensor_type).to(device=target_device, dtype=dtype)
            if self.model_sd_ops is not None:
                # apply_to_key_value may produce additional keys from a single tensor
                # (e.g. GEMMA_LLM_KEY_OPS duplicates embed_tokens.weight → lm_head.weight
                # so that model.generate() works for prompt enhancement).
                for out_key, out_val in self.model_sd_ops.apply_to_key_value(model_key, t_bf16):
                    non_block_sd[out_key] = out_val
            else:
                non_block_sd[model_key] = t_bf16
        del _tensor_map, _reader
        meta_model.load_state_dict(non_block_sd, strict=False, assign=True)
        del non_block_sd

        # --- Wire up GPU-direct GGUF block streaming ---
        # Blocks are dequantized directly on GPU (no CPU pinned buffer needed).
        # copy_stream sequences: GPU dequant → GPU→GPU copy → compute.
        copy_stream = torch.cuda.Stream(device=target_device)

        direct_reader = GGUFDirectGPUReader(self.model_path, block_key_map, target_device)

        # Source GPU pool: holds freshly dequantized blocks (replaces pinned CPU pool)
        src_gpu_pool = WeightPool(
            block_layout,
            cpu_slots_count,
            target_device,
            reuse_barrier=lambda event: copy_stream.wait_event(event),
            pin_memory=False,
        )
        source = GGUFGPUWeightSource(src_gpu_pool, direct_reader, copy_stream)

        # Compute GPU pool: double-buffers the block active during forward pass
        gpu_pool = WeightPool(
            block_layout,
            gpu_slots_count,
            target_device,
            reuse_barrier=lambda event: copy_stream.wait_event(event),
        )
        provider = WeightsProvider(
            gpu_pool,
            copy_stream,
            target_device,
            source,
            [],
            self.blocks_prefix,
            fuse_rule=self.fuse_rule,
        )
        return BlockStreamingWrapper(
            model=meta_model,
            blocks=blocks,
            provider=provider,
            target_device=target_device,
        )


def _move_cpu_buffers(module: nn.Module, device: torch.device, skip_child: str = "") -> None:
    """Move non-meta CPU buffers to *device*, skipping the named direct child."""
    for buf_name, buf in module._buffers.items():
        if buf is not None and not buf.is_meta and buf.device.type == "cpu":
            module._buffers[buf_name] = buf.to(device)
    for child_name, child in module._modules.items():
        if child_name == skip_child:
            continue
        _move_cpu_buffers(child, device)


def _patch_gemma_device(wrapper: BlockStreamingWrapper, device: torch.device) -> None:
    """Fix two issues that arise when streaming Gemma from a language-only GGUF.

    Issue 1 — wrong ``model.device``:
        HuggingFace's ``PreTrainedModel.device`` returns the device of the
        *first* parameter in the module tree, which is the vision tower
        (still on ``meta`` because the language-only GGUF has no vision
        weights).  ``GemmaTextEncoder.encode()`` uses ``self.model.device`` to
        place ``input_ids`` / ``attention_mask``, so this causes a
        "Cannot copy out of meta tensor" crash.  We override ``device`` with a
        dynamic subclass that always returns the real compute device.

    Issue 2 — CPU buffers in the language model:
        ``GEMMA_MODEL_OPS`` (``create_and_populate``) registers ``embed_scale``
        and RoPE ``inv_freq`` buffers on CPU because it runs while the module
        is still being built.  When the GPU forward pass executes, these CPU
        buffers cause device-mismatch errors.  We move all non-meta CPU buffers
        in the language model's non-block submodules to the compute device.
        Transformer block buffers are skipped — they are managed by the
        streaming hooks and live in pinned CPU RAM until pulled to GPU.
    """
    try:
        gemma_enc = wrapper._model          # GemmaTextEncoder
        hf_model = gemma_enc.model          # Gemma3ForConditionalGeneration

        # Fix 1: patch device property
        _fixed_device = device

        class _DeviceFixed(type(hf_model)):
            @property
            def device(self):  # noqa: N805
                return _fixed_device

        hf_model.__class__ = _DeviceFixed

        # Fix 2: move CPU buffers (embed_scale, inv_freq, …) to compute device,
        # skipping the transformer blocks ("layers") which are streaming-managed.
        l_model = hf_model.model.language_model
        _move_cpu_buffers(l_model, device, skip_child="layers")

    except (AttributeError, TypeError):
        pass


@dataclass(frozen=True)
class GemmaGGUFStreamingModelBuilder(GGUFStreamingModelBuilder[ModelType]):
    """CPU block-streaming builder for Gemma 3 GGUF checkpoints (llama.cpp format).

    Identical to :class:`GGUFStreamingModelBuilder` but applies the
    llama.cpp-to-HuggingFace key mapping (``blk.N.attn_q.weight`` →
    ``language_model.model.layers.N.self_attn.q_proj.weight``) before
    ``model_sd_ops`` so that ``GEMMA_LLM_KEY_OPS`` works unchanged.

    Usage::

        from ltx_core.text_encoders.gemma.encoders.encoder_configurator import (
            GemmaTextEncoderConfigurator, GEMMA_LLM_KEY_OPS, GEMMA_MODEL_OPS,
        )
        from ltx_core.text_encoders.gemma.encoders.base_encoder import (
            module_ops_from_gemma_root,
        )

        module_ops = module_ops_from_gemma_root("./gemma-tokenizer/")
        builder = GemmaGGUFStreamingModelBuilder(
            model_path="gemma-3-12b-it-qat-UD-Q4_K_XL.gguf",
            model_class_configurator=GemmaTextEncoderConfigurator,
            model_sd_ops=GEMMA_LLM_KEY_OPS,
            module_ops=(GEMMA_MODEL_OPS, *module_ops),
            blocks_attr="model.model.language_model.layers",
            blocks_prefix="model.model.language_model.layers",
        )
    """

    @property
    def _loader(self):
        from ltx_core.text_encoders.gemma.gguf_loader import GemmaGGUFStateDictLoader
        return GemmaGGUFStateDictLoader()

    @property
    def model_loader(self):
        from ltx_core.text_encoders.gemma.gguf_loader import GemmaGGUFStateDictLoader
        return GemmaGGUFStateDictLoader()

    def _scan_keys(
        self,
        path: str,
        sd_ops: SDOps | None,
        blocks_prefix: str,
    ) -> tuple[dict[int, list[tuple[str, str]]], list[tuple[str, str]]]:
        from ltx_core.text_encoders.gemma.gguf_loader import gemma_gguf_key_to_hf
        return _scan_gguf_keys(path, sd_ops, blocks_prefix, key_mapper=gemma_gguf_key_to_hf)

    def build(
        self,
        target_device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        cpu_slots_count: int | None = None,
        gpu_slots_count: int | None = None,
        **_kwargs: object,
    ) -> BlockStreamingWrapper:
        wrapper = super().build(
            target_device=target_device,
            dtype=dtype,
            cpu_slots_count=cpu_slots_count,
            gpu_slots_count=gpu_slots_count,
        )
        _device = target_device if target_device is not None else torch.device("cuda")
        _patch_gemma_device(wrapper, _device)
        return wrapper
