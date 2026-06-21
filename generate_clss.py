"""
generate_clss.py  —  Long-video generation via CLSS with 16 GB VRAM optimisations.
                     Loads the transformer and Gemma text encoder from GGUF files.

Usage
-----
    python generate_clss.py \
        --gguf-path        ltx-2.3-22b-dev-UD-Q4_K_S.gguf \
        --embeddings-path  ltx-2.3-22b-dev_embeddings_connectors.safetensors \
        --audio-vae-path   ltx-2.3-22b-dev_audio_vae.safetensors \
        --video-vae-path   ltx-2.3-22b-dev_video_vae.safetensors \
        --gemma-gguf       gemma-3-12b-it-qat-UD-Q4_K_XL.gguf \
        --gemma-tokenizer  ./gemma-tokenizer/ \
        --prompt           "A red fox trots through a snowy pine forest at dusk."

Gemma tokenizer
---------------
The GGUF file contains only the model weights; the tokenizer must be supplied
as a separate directory.  Download just the tokenizer files from HuggingFace
(~15 MB total — no full model download needed):

    huggingface-cli download google/gemma-3-12b-it \\
        --include "tokenizer*" "special_tokens_map.json" "preprocessor_config.json" \\
        --local-dir ./gemma-tokenizer/

16 GB VRAM strategy
--------------------
The LTX-2.3 22B transformer is ~44 GB in BF16.  Three settings together keep it
inside 16 GB VRAM while maintaining acceptable speed:

  1. OffloadMode.CPU        — Transformer blocks live in pinned CPU RAM.  Only the
                              2 blocks needed for the current layer-streaming step
                              reside on GPU at a time (~5 GB for model weights).
                              Requires ~48 GB free RAM for the full dequantized model.

  2. max_batch_size=1       — CFG/STG guidance passes are serialised instead of
                              batched, reducing peak activation memory by 2–4×.

  3. CLSS streaming         — Generates video in short temporal chunks, keeping
                              latent memory O(overlap) instead of O(total length).

Combined peak VRAM at 480×704:
  ~5 GB  model blocks (two blocks, BF16, block-streamed)
  ~4 GB  activations + latents
  ~2 GB  VAE decoder, text encoder (loaded/freed one at a time)
  ≈ 11 GB peak, leaving ~5 GB headroom on a 16 GB card.

RAM requirement
---------------
The Q4_K_S GGUF (~11 GB on disk) is dequantized to BF16 (~44 GB) in pinned CPU
RAM before streaming blocks to the GPU.  Make sure the system has ≥ 48 GB of
free RAM.  To avoid the dequantization overhead on every run, use the companion
script to pre-convert once:

    python convert_gguf.py \\
        --input  ltx-2.3-22b-dev-UD-Q4_K_S.gguf \\
        --output ./ltx-2.3-22b-bf16/

Then pass --checkpoint-path ./ltx-2.3-22b-bf16/ (without --gguf-path).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Must be set before CUDA initializes. Reduces allocator fragmentation —
# we're running close to the 16 GB ceiling (model blocks ~5 GB + per-chunk
# activations), and the stock allocator was failing to allocate ~700 MiB
# while several hundred MiB sat "reserved but unallocated" from fragmentation.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch

# ---------------------------------------------------------------------------
# Path setup — allow running from the repo root without installing packages
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent
for _pkg in ("ltx-core", "ltx-pipelines"):
    _src = _REPO_ROOT / "packages" / _pkg / "src"
    if _src.exists() and str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from ltx_core.block_streaming.gguf_builder import (
    GemmaGGUFStreamingModelBuilder,
    GGUFStreamingModelBuilder,
)

from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.loader.gguf_loader import CombinedStateDictLoader, GGUFStateDictLoader
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
from ltx_core.loader.sd_ops import SDOps
from ltx_core.loader.sft_loader import SafetensorsStateDictLoader
from ltx_core.model.transformer import LTXModelConfigurator
from ltx_core.text_encoders.gemma import (
    GEMMA_LLM_KEY_OPS,
    GEMMA_MODEL_OPS,
    EmbeddingsProcessorConfigurator,
    GemmaTextEncoderConfigurator,
    module_ops_from_gemma_root,
)

# GGUF exports use bare keys without the "model.diffusion_model." prefix that
# EMBEDDINGS_PROCESSOR_KEY_OPS expects for connector weights.
_GGUF_EMBEDDINGS_PROCESSOR_KEY_OPS = (
    SDOps("GGUF_EMBEDDINGS_PROCESSOR_KEY_OPS")
    .with_matching(prefix="text_embedding_projection.aggregate_embed.")
    .with_replacement("text_embedding_projection.aggregate_embed.", "feature_extractor.aggregate_embed.")
    .with_matching(prefix="text_embedding_projection.video_aggregate_embed.")
    .with_replacement("text_embedding_projection.video_aggregate_embed.", "feature_extractor.video_aggregate_embed.")
    .with_matching(prefix="text_embedding_projection.audio_aggregate_embed.")
    .with_replacement("text_embedding_projection.audio_aggregate_embed.", "feature_extractor.audio_aggregate_embed.")
    # standard safetensors format (with prefix)
    .with_matching(prefix="model.diffusion_model.video_embeddings_connector.")
    .with_replacement("model.diffusion_model.video_embeddings_connector.", "video_connector.")
    .with_matching(prefix="model.diffusion_model.audio_embeddings_connector.")
    .with_replacement("model.diffusion_model.audio_embeddings_connector.", "audio_connector.")
    # GGUF-native format (no model.diffusion_model. prefix)
    .with_matching(prefix="video_embeddings_connector.")
    .with_replacement("video_embeddings_connector.", "video_connector.")
    .with_matching(prefix="audio_embeddings_connector.")
    .with_replacement("audio_embeddings_connector.", "audio_connector.")
)

from ltx_core.model.video_vae import (
    MEMORY_EFFICIENT_DECODE,
    VideoDecoderConfigurator,
    VideoEncoderConfigurator,
    TilingConfig,
    TemporalTilingConfig,
    get_video_chunks_number,
)
from ltx_pipelines.streaming import CLSSConfig, CLSSStreamingPipeline
from ltx_pipelines.streaming.pipeline import build_chunk_schedule, _pixel_to_latent_frames
from ltx_pipelines.utils.args import CompileAction
from ltx_pipelines.utils.blocks import ImageConditioner, VideoDecoder
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.types import OffloadMode

# The standalone video VAE safetensors file uses bare key prefixes:
#   decoder.*           → *            (strip decoder. prefix)
#   per_channel_statistics.*  → per_channel_statistics.*  (keep)
# The standard VAE_DECODER_COMFY_KEYS_FILTER expects "vae.decoder." which doesn't exist here.
_VIDEO_VAE_BARE_KEYS = (
    SDOps("VIDEO_VAE_BARE_KEYS")
    .with_matching(prefix="decoder.")
    .with_matching(prefix="per_channel_statistics.")
    .with_replacement("decoder.", "")
)

# Same file, encoder side — used by ImageConditioner when --image is passed.
# VAE_ENCODER_COMFY_KEYS_FILTER expects "vae.encoder." which doesn't exist here.
_VIDEO_VAE_BARE_ENCODER_KEYS = (
    SDOps("VIDEO_VAE_BARE_ENCODER_KEYS")
    .with_matching(prefix="encoder.")
    .with_matching(prefix="per_channel_statistics.")
    .with_replacement("encoder.", "")
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generate_clss")


# ---------------------------------------------------------------------------
# Defaults tuned for 16 GB VRAM + LTX-2.3
# ---------------------------------------------------------------------------

_DEFAULT_HEIGHT  = 512           # px  (must be divisible by 32)
_DEFAULT_WIDTH   = 768           # px  (must be divisible by 32)
_DEFAULT_CHUNKS  = 25             # number of CLSS chunks; frames = chunks × new_lf × 8
_DEFAULT_FPS     = 25.0
_DEFAULT_STEPS   = 30
_DEFAULT_SEED    = 42

# Multi-scene test prompts — one per chunk, scenes evolve over the video.
# Override on the command line with --prompts 'scene 1' 'scene 2' ...
_DEFAULT_PROMPTS = [
    "3D, POV driving a sleek Lotus-style sports car along the sunny Malibu coastline, golden sunlight reflecting off the ocean, palm trees passing by, smooth winding roads, vibrant summer colors, cinematic motion, ultra-realistic detailing, high-end luxury car aesthetic, 8K resolution."
]

# CLSS defaults — tuned for 16 GB VRAM with LTX-2.3 GGUF
# tau_c=0.05: strong overlap constraint (0.20 gave cosine_sim=0.55 at chunk-3).
# overlap=8: 8 latent frames ≈ 57 pixel frames of hard context from the previous
#   chunk.  With tau_c=0.05 that's nearly frozen — the model sees clean prior-chunk
#   frames at the start of each new chunk, giving "image 1-3 are from old chunk"
#   style continuity.  Cost: ~17% more denoising steps per chunk vs overlap=4.
# beta=0.25, ema_lambda=0.10: gentle AdaIN correction — 25% blend towards EMA.
# 0.50 was too aggressive: EMA is dominated by chunk-0 stats (~80% weight after 3 chunks)
# so beta=0.50 pulled each chunk strongly back towards the first chunk's look, causing
# the last frames (farthest from any frozen reference) to look "out of touch".
# freq_gamma conservative: (0.5,0.8) attenuated 13-14% per chunk; now (0.2,0.3).
_DEFAULT_TAU_C   = 0.05
_DEFAULT_BETA    = 0.40
_DEFAULT_LAMBDA  = 0.10
_DEFAULT_OVERLAP = 8
_DEFAULT_NEW_LF  = 13
# 21 gave intra_chunk_sim=0.03–0.11 (near-random — scene drifts completely within the chunk).
# 13 lf ≈ 97 px ≈ 4 s @24 fps — safer against intra-chunk collapse.

# Guidance defaults for LTX-2.3
# CFG=4.5 (instead of 3.0) more aggressively suppresses negative-prompt artifacts
# (text watermarks, watermark logos) that the 4-bit model tends to hallucinate
# in later chunks.  Use --video-cfg 3.0 if you want the standard setting.
#
# STG and modality guidance are OFF by default for speed:
#   With STG=0, modality=0, max_batch_size=2 → B=2 CFG passes in 1 transformer call.
#   With STG=1, modality=3, max_batch_size=1 → B=4 in 4 serial calls (4× slower).
#   Speed impact: 4× more block-streaming rounds per step.
#   Enable with --video-stg 1.0 --modality-scale 3.0 for highest quality.
_DEFAULT_VIDEO_CFG     = 4.5
_DEFAULT_VIDEO_STG     = 1.0   # 1.0 adds +1 PTB pass (4× slower); enable for quality
_DEFAULT_VIDEO_RESCALE = 0.7
_DEFAULT_MODALITY      = 3.0   # 3.0 adds +1 MOD pass (4× slower); enable for quality
_DEFAULT_STG_BLOCKS    = [28]
_DEFAULT_AUDIO_CFG     = 7.0
_DEFAULT_AUDIO_STG     = 1.0   # 1.0 adds audio PTB pass; same overhead as video STG
_DEFAULT_AUDIO_RESCALE = 0.7


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vram_gb() -> float | None:
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / 1024**3
    return None


def _log_image_resize(path: str, target_h: int, target_w: int) -> None:
    """Log how the conditioning image will be resized to match the target resolution.

    Mirrors resize_and_center_crop geometry (scale-to-fill → center crop) so the
    user can see how much of their image will be discarded before waiting for a
    100-minute generation run.
    """
    import math
    from PIL import Image as PilImage
    with PilImage.open(path) as img:
        src_w, src_h = img.size  # PIL: (width, height)
    target_ar = target_w / target_h
    src_ar    = src_w    / src_h
    if src_h == target_h and src_w == target_w:
        logger.info("  Image        : %s  (%dx%d — exact match, no resize)", path, src_w, src_h)
        return
    scale = max(target_h / src_h, target_w / src_w)
    new_h = math.ceil(src_h * scale)
    new_w = math.ceil(src_w * scale)
    crop_top  = (new_h - target_h) // 2
    crop_left = (new_w - target_w) // 2
    pct_h = (new_h - target_h) / new_h * 100
    pct_w = (new_w - target_w) / new_w * 100
    logger.info(
        "  Image        : %s  %dx%d (AR %.3f) → scale ×%.3f → %dx%d "
        "→ center-crop(top=%d,left=%d) → %dx%d (AR %.3f)",
        path, src_w, src_h, src_ar, scale, new_w, new_h,
        crop_left, crop_top, target_w, target_h, target_ar,
    )
    if pct_h > 15 or pct_w > 15:
        logger.warning(
            "Image aspect ratio mismatch: %.0f%% of height or %.0f%% of width will be "
            "cropped away.  Pre-crop your image to %.2f:1 (W:H) for best results.",
            pct_h, pct_w, target_ar,
        )


def _print_config_summary(args: argparse.Namespace, clss: CLSSConfig, num_frames: int) -> None:
    vram = _vram_gb()
    total_lf = _pixel_to_latent_frames(num_frames)
    chunks = build_chunk_schedule(total_lf, clss)
    rho = (1.0 - clss.beta) * (1.0 - clss.tau_c)

    logger.info("=" * 64)
    logger.info("LTX-2.3  22B  —  CLSS streaming generation (GGUF)")
    logger.info("=" * 64)
    logger.info("  Transformer  : %s", args.gguf_path)
    logger.info("  Embeddings   : %s", args.embeddings_path)
    logger.info("  Video VAE    : %s", args.video_vae_path)
    logger.info("  Audio VAE    : %s", args.audio_vae_path)
    logger.info("  Gemma GGUF   : %s", args.gemma_gguf)
    logger.info("  Gemma tokens : %s", args.gemma_tokenizer)
    logger.info("  Resolution   : %dx%d @ %.0f fps", args.width, args.height, args.fps)
    logger.info("  Frames       : %d px  →  %d latent", num_frames, total_lf)
    logger.info("  Steps        : %d", args.steps)
    logger.info("  Seed         : %d", args.seed)
    logger.info("  GPU VRAM     : %s", f"{vram:.1f} GB" if vram else "CPU only")
    logger.info("  Offload mode : CPU block streaming (pinned RAM)")
    logger.info("  Guidance     : video_cfg=%.1f  stg=%.1f  audio_cfg=%.1f  modality=%.1f  "
                "batch_size=%d  [passes/step=%d]",
                args.video_cfg, args.video_stg, args.audio_cfg, args.modality_scale,
                args.batch_size,
                max(1, (2 + (1 if args.video_stg != 0 else 0) + (1 if args.modality_scale != 0 else 0))
                    // args.batch_size))
    logger.info(
        "  CLSS chunks  : %d  (τc=%.2f  β=%.2f  λ=%.2f  σ_max_drift=%.2f  "
        "anchor_force=%d  overlap=%d  new=%d lf)",
        len(chunks), clss.tau_c, clss.beta, clss.ema_lambda, clss.ema_sigma_max_drift,
        clss.anchor_force_every, clss.overlap_latent_frames, clss.new_latent_frames,
    )
    logger.info("  ρ_loop est.  : %.3f  (stable when < 1.0)", rho)
    logger.info("  Prompts      : %d  (last repeats for remaining chunks)", len(args.prompts))
    for i, pr in enumerate(args.prompts):
        logger.info("    [%d] %s", i + 1, pr[:120] + ("…" if len(pr) > 120 else ""))
    logger.info("  Output       : %s", args.output)
    logger.info("=" * 64)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CLSS streaming video generation — GGUF transformer, 16 GB VRAM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Model files ---
    p.add_argument("--gguf-path", required=True,
                   help="Path to the transformer GGUF file "
                        "(e.g. ltx-2.3-22b-dev-UD-Q4_K_S.gguf).")
    p.add_argument("--embeddings-path", required=True,
                   help="Path to ltx-2.3-22b-dev_embeddings_connectors.safetensors.")
    p.add_argument("--audio-vae-path", required=True,
                   help="Path to ltx-2.3-22b-dev_audio_vae.safetensors.")
    p.add_argument("--video-vae-path", required=True,
                   help="Path to ltx-2.3-22b-dev_video_vae.safetensors.")
    p.add_argument("--gemma-gguf", required=True,
                   help="Path to the Gemma 3 GGUF file "
                        "(e.g. gemma-3-12b-it-qat-UD-Q4_K_XL.gguf).")
    p.add_argument("--gemma-tokenizer", required=True,
                   help="Directory containing Gemma tokenizer files "
                        "(tokenizer.model / tokenizer.json + preprocessor_config.json). "
                        "Download with: huggingface-cli download google/gemma-3-12b-it "
                        "--include 'tokenizer*' 'special_tokens_map.json' "
                        "'preprocessor_config.json' --local-dir ./gemma-tokenizer/")

    # --- Prompts ---
    p.add_argument("--prompts", nargs="+", default=_DEFAULT_PROMPTS,
                   metavar="PROMPT",
                   help="Per-chunk text prompts.  Pass one string per chunk; the last "
                        "entry repeats for any remaining chunks.  Single-prompt example: "
                        "--prompts 'A fox trots through a forest.'  "
                        "Multi-scene: --prompts 'Scene 1 …' 'Scene 2 …' 'Scene 3 …'")
    p.add_argument("--negative-prompt", default=(
        "blurry, low quality, distorted, flickering, cartoon, unrealistic, "
        "watermark, text overlay, text on screen, text in frame, captions, "
        "subtitles, logo, title card, stock footage watermark"
    ))
    p.add_argument("--enhance-prompt", action="store_true",
                   help=(
                       "Use Gemma to automatically expand the first prompt. "
                       "NOTE: not compatible with --gemma-gguf (block streaming) — "
                       "model.generate() with block streaming requires ~60 min for 512 tokens "
                       "and will be silently skipped. Only works without streaming."
                   ))

    # --- Resolution / duration ---
    p.add_argument("--height",      type=int,   default=_DEFAULT_HEIGHT)
    p.add_argument("--width",       type=int,   default=_DEFAULT_WIDTH)
    p.add_argument("--num-chunks",  type=int,   default=_DEFAULT_CHUNKS,
                   help="Number of CLSS chunks to generate.  "
                        "Total pixel frames = num_chunks × new_latent_frames × 8.")
    p.add_argument("--fps",         type=float, default=_DEFAULT_FPS)

    # --- Diffusion ---
    p.add_argument("--steps",      type=int,   default=_DEFAULT_STEPS,
                   help="Denoising steps per chunk (default 30). 15 is 2× faster with "
                        "minor quality drop; below 10 degrades noticeably.")
    p.add_argument("--compile", nargs="*", default=None, action=CompileAction,
                   help="Enable torch.compile for transformer blocks. "
                        "Adds ~2-3 min warmup on first chunk; subsequent chunks are 15-25%% faster. "
                        "Incompatible with --offload-mode (block-streaming). "
                        "Pass alone for defaults, or KEY=VALUE to override: "
                        "--compile mode=reduce-overhead")
    p.add_argument("--seed",       type=int,   default=_DEFAULT_SEED)
    p.add_argument("--batch-size", type=int,   default=1,
                   help="Max guidance-pass batch size.  2=default: batches CFG cond+uncond "
                        "into one transformer call, halving block-streaming rounds.  "
                        "Use 1 if you hit OOM.")

    # --- Guidance ---
    p.add_argument("--video-cfg",  type=float, default=_DEFAULT_VIDEO_CFG,
                   help="Video CFG scale.  4.5=default (suppresses 4-bit text artifacts); "
                        "3.0=LTX standard (may allow watermark hallucinations in later chunks).")
    p.add_argument("--video-stg",  type=float, default=_DEFAULT_VIDEO_STG,
                   help="Video STG scale.  0.0=default (off, fast); 1.0=on (adds +1 transformer "
                        "pass per step — ~25%% slower with batch-size=2, 4× slower with "
                        "batch-size=1).  Only useful when you can afford the compute cost.")
    p.add_argument("--audio-stg",  type=float, default=_DEFAULT_AUDIO_STG,
                   help="Audio STG scale.  0.0=default (off).  See --video-stg notes.")
    p.add_argument("--modality-scale", type=float, default=_DEFAULT_MODALITY,
                   help="Cross-modal guidance scale.  0.0=default (off, fast); 3.0=on (adds +1 "
                        "transformer pass per step — same overhead as STG).  Helps align video "
                        "and audio content when enabled.")
    p.add_argument("--audio-cfg",  type=float, default=_DEFAULT_AUDIO_CFG)

    # --- CLSS ---
    g = p.add_argument_group("CLSS closed-loop corrections")
    g.add_argument("--clss-tau-c",      type=float, default=_DEFAULT_TAU_C,
                   help="Overlap re-noising τc (§2.1).  0=hardest constraint/most continuity, "
                        "0.05=recommended, 0.20=paper value (too loose for streaming 22B).")
    g.add_argument("--clss-beta",       type=float, default=_DEFAULT_BETA,
                   help="AdaIN drift correction β (§2.3).  0=off, 0.40=recommended, 0.5=aggressive "
                        "(pulls too hard towards EMA dominated by chunk-0 stats).")
    g.add_argument("--clss-overlap",    type=int,   default=_DEFAULT_OVERLAP,
                   help="SLB overlap in latent frames.  4=default (~0.8s).  "
                        "8 gives better continuity at ~30%% more compute per chunk.")
    g.add_argument("--clss-lambda",     type=float, default=_DEFAULT_LAMBDA,
                   help="EMA update rate λ (§2.3/§2.4).  0.05=slow/stable, 0.10=faster drift correction.")
    g.add_argument("--clss-sigma-max-drift", type=float, default=0.05,
                   help="Max fractional growth of per-channel EMA σ relative to chunk-0 σ (§2.3). "
                        "0.05 = EMA std may drift at most 5%% above its initial value, preventing "
                        "AdaIN from amplifying late chunks.  0 = uncapped (old behaviour).")
    g.add_argument("--clss-anchor-force-every", type=int, default=5,
                   help="Force a new anchor into the bank every N chunks regardless of similarity. "
                        "Prevents the bank from stagnating on chunk-0 frames when the scene is "
                        "visually stable (similarity stays above threshold, persistence never fires). "
                        "0 = disabled (similarity-only anchoring).")
    g.add_argument("--clss-new-frames", type=int,   default=_DEFAULT_NEW_LF,
                   help="New latent frames per chunk (13 lf ≈ 97 px ≈ 4 s @24 fps). "
                        "Smaller = less intra-chunk drift but more chunks. "
                        "21 caused intra_chunk_sim=0.03 (scene collapses); 13 is safer.")
    g.add_argument("--no-measure-g", dest="measure_g", action="store_false", default=True,
                   help="Disable per-chunk g measurement (§3.7). By default, every chunk with "
                        "overlap runs a second perturbed denoising pass to measure g and "
                        "ρ_closed = g·ρ_loop. Doubles denoising time per chunk. "
                        "Pass --no-measure-g to skip.")

    # --- I/O ---
    p.add_argument("--output", default="output_clss.mp4")
    p.add_argument("--image",  default=None,
                   help="Optional conditioning image path for the first frame.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Logging verbosity. DEBUG adds per-channel latent stats, "
                        "band energy gains, and anchor similarity per chunk.")
    p.add_argument("--checkpoint-dir", default=None,
                   help="Directory for per-chunk JSONL research logs and periodic "
                        "latent checkpoints. Defaults to '<output stem>_checkpoints'.")
    p.add_argument("--checkpoint-every", type=int, default=5,
                   help="Save a cumulative latent checkpoint every N chunks "
                        "(only when there are more than N chunks total).")

    return p.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # ------------------------------------------------------------------
    # Validate paths
    # ------------------------------------------------------------------
    for flag, path in [
        ("--gguf-path",        args.gguf_path),
        ("--embeddings-path",  args.embeddings_path),
        ("--audio-vae-path",   args.audio_vae_path),
        ("--video-vae-path",   args.video_vae_path),
        ("--gemma-gguf",       args.gemma_gguf),
        ("--gemma-tokenizer",  args.gemma_tokenizer),
    ]:
        if not Path(path).exists():
            sys.exit(f"ERROR: {flag} path not found: {path}")

    if not args.gguf_path.endswith(".gguf"):
        logger.warning(
            "--gguf-path does not end in .gguf: %s — make sure it is a GGUF file.",
            args.gguf_path,
        )
    if not args.gemma_gguf.endswith(".gguf"):
        logger.warning(
            "--gemma-gguf does not end in .gguf: %s — make sure it is a GGUF file.",
            args.gemma_gguf,
        )

    # ------------------------------------------------------------------
    # VRAM guard
    # ------------------------------------------------------------------
    vram = _vram_gb()
    if vram is not None and vram < 14.0:
        logger.warning(
            "Detected %.1f GB VRAM.  This script targets ≥ 16 GB.  "
            "Consider --height 360 --width 640 or --clss-new-frames 12.",
            vram,
        )

    # ------------------------------------------------------------------
    # CLSS config
    # ------------------------------------------------------------------
    clss_config = CLSSConfig(
        tau_c=args.clss_tau_c,
        beta=args.clss_beta,
        ema_lambda=args.clss_lambda,
        ema_sigma_max_drift=args.clss_sigma_max_drift,
        anchor_force_every=args.clss_anchor_force_every,
        overlap_latent_frames=args.clss_overlap,
        new_latent_frames=args.clss_new_frames,
        measure_g=args.measure_g,
    )

    # Derive total pixel frames from chunk count.
    # Each chunk contributes new_latent_frames latent frames; total latent
    # frames = num_chunks × new_latent_frames; pixel frames = (lf-1)×8+1.
    num_frames = (args.num_chunks * clss_config.new_latent_frames - 1) * 8 + 1

    _print_config_summary(args, clss_config, num_frames)

    # ------------------------------------------------------------------
    # Build GGUF-backed transformer builders
    #
    # GGUFStateDictLoader  — dequantizes Q4_K_S → BF16 at load time.
    # SingleGPUModelBuilder — used by DiffusionStage for non-streaming path.
    # GGUFStreamingModelBuilder — CPU block-streaming path for 16 GB VRAM:
    #   all transformer blocks pinned in CPU RAM (~44 GB BF16),
    #   only 2 blocks on GPU at a time (~10 GB peak GPU).
    # ------------------------------------------------------------------
    logger.info("Configuring GGUF transformer builders …")
    gguf_loader = GGUFStateDictLoader()

    # GGUF exports already use bare key names (no "model.diffusion_model." prefix),
    # so LTXV_MODEL_COMFY_RENAMING_MAP must NOT be used — pass sd_ops=None.
    gguf_single_builder = SingleGPUModelBuilder(
        model_path=args.gguf_path,
        model_class_configurator=LTXModelConfigurator,
        model_sd_ops=None,
        model_loader=gguf_loader,
    )

    gguf_streaming_builder = GGUFStreamingModelBuilder(
        model_path=args.gguf_path,
        model_class_configurator=LTXModelConfigurator,
        model_sd_ops=None,
        blocks_attr="transformer_blocks",
        blocks_prefix="transformer_blocks",
    )

    # ------------------------------------------------------------------
    # Embeddings processor builder
    #
    # The EmbeddingsProcessor weights come from two sources:
    #   - GGUF: video_embeddings_connector.* and audio_embeddings_connector.*
    #           (bare keys — no model.diffusion_model. prefix in GGUF exports)
    #   - embeddings_connectors.safetensors: feature_extractor projection weights
    #           (text_embedding_projection.{video,audio}_aggregate_embed.*)
    #
    # CombinedStateDictLoader merges tensors from both, using GGUF config for
    # architecture construction (connector_num_layers=8, etc.).
    # ------------------------------------------------------------------
    embeddings_combined_loader = CombinedStateDictLoader([
        (args.gguf_path,        GGUFStateDictLoader()),
        (args.embeddings_path,  SafetensorsStateDictLoader()),
    ])
    embeddings_builder = SingleGPUModelBuilder(
        model_path=args.gguf_path,
        model_class_configurator=EmbeddingsProcessorConfigurator,
        model_sd_ops=_GGUF_EMBEDDINGS_PROCESSOR_KEY_OPS,
        model_loader=embeddings_combined_loader,
    )

    # ------------------------------------------------------------------
    # Gemma GGUF streaming builder
    #
    # GemmaGGUFStreamingModelBuilder translates llama.cpp GGUF tensor names
    # (blk.N.attn_q.weight …) to HuggingFace format before GEMMA_LLM_KEY_OPS,
    # then streams Gemma transformer blocks from pinned CPU RAM to GPU.
    #
    # module_ops_from_gemma_root loads the tokenizer and image processor from
    # the --gemma-tokenizer directory (tiny files, no full model needed).
    # ------------------------------------------------------------------
    logger.info("Configuring Gemma GGUF builder …")
    gemma_module_ops = module_ops_from_gemma_root(args.gemma_tokenizer)
    gemma_streaming_builder = GemmaGGUFStreamingModelBuilder(
        model_path=args.gemma_gguf,
        model_class_configurator=GemmaTextEncoderConfigurator,
        model_sd_ops=GEMMA_LLM_KEY_OPS,
        module_ops=(GEMMA_MODEL_OPS, *gemma_module_ops),
        blocks_attr="model.model.language_model.layers",
        blocks_prefix="model.model.language_model.layers",
    )

    # ------------------------------------------------------------------
    # Build pipeline
    # ------------------------------------------------------------------
    logger.info("Building CLSSStreamingPipeline …")
    pipeline = CLSSStreamingPipeline(
        checkpoint_path=args.gguf_path,   # used only for fallback / logging
        gemma_root=args.gemma_tokenizer,  # unused — streaming_text_encoder_builder takes over
        loras=[],
        offload_mode=OffloadMode.CPU,
        compilation_config=args.compile,
        # Per-component paths (separate safetensors files)
        embeddings_path=args.embeddings_path,
        video_vae_path=args.video_vae_path,
        audio_vae_path=args.audio_vae_path,
        # GGUF-backed transformer builders
        transformer_builder=gguf_single_builder,
        streaming_builder=gguf_streaming_builder,
        # GGUF-backed Gemma text encoder
        streaming_text_encoder_builder=gemma_streaming_builder,
    )

    # Fix the embeddings processor builder to use both GGUF + safetensors
    # (the default builder from CLSSStreamingPipeline only reads the safetensors
    # file, which lacks the connector architecture and weights)
    pipeline.prompt_encoder._embeddings_processor_builder = embeddings_builder

    # Fix video VAE decoder: standalone safetensors uses bare "decoder.*" keys,
    # not the "vae.decoder.*" prefix that VAE_DECODER_COMFY_KEYS_FILTER expects.
    pipeline.video_decoder = VideoDecoder(
        checkpoint_path=args.video_vae_path,
        dtype=pipeline.dtype,
        device=pipeline.device,
        decoder_builder=SingleGPUModelBuilder(
            model_path=args.video_vae_path,
            model_class_configurator=VideoDecoderConfigurator,
            model_sd_ops=_VIDEO_VAE_BARE_KEYS,
            module_ops=(MEMORY_EFFICIENT_DECODE,),
        ),
    )

    # Fix image conditioner encoder: same file uses bare "encoder.*" keys,
    # not the "vae.encoder.*" prefix that VAE_ENCODER_COMFY_KEYS_FILTER expects.
    pipeline.image_conditioner._encoder_builder = SingleGPUModelBuilder(
        model_path=args.video_vae_path,
        model_class_configurator=VideoEncoderConfigurator,
        model_sd_ops=_VIDEO_VAE_BARE_ENCODER_KEYS,
    )

    # ------------------------------------------------------------------
    # Image conditioning (optional)
    # ------------------------------------------------------------------
    images = []
    if args.image:
        from ltx_pipelines.utils.args import ImageConditioningInput
        images = [ImageConditioningInput(path=args.image, frame_idx=0, strength=1.0)]
        _log_image_resize(args.image, args.height, args.width)

    # ------------------------------------------------------------------
    # VAE temporal tiling — required for long videos (>~60 latent frames)
    # because the full latent is too large to decode in a single GPU pass.
    # tile_size_in_frames is in *pixel* frames; the VAE divides it by 8
    # internally to get latent frame tile size.
    # 128 px frames = 16 latent frames per tile → well within 16 GB VRAM.
    # ------------------------------------------------------------------
    tiling_config = TilingConfig(
        temporal_config=TemporalTilingConfig(
            tile_size_in_frames=128,
            tile_overlap_in_frames=32,
        )
    )

    # ------------------------------------------------------------------
    # Run generation
    # ------------------------------------------------------------------
    checkpoint_dir = args.checkpoint_dir or f"{Path(args.output).stem}_checkpoints"
    logger.info("Starting CLSS streaming generation …")
    video_iter, audio = pipeline(
        prompts=args.prompts,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=num_frames,
        frame_rate=args.fps,
        num_inference_steps=args.steps,
        video_guider_params=MultiModalGuiderParams(
            cfg_scale=args.video_cfg,
            stg_scale=args.video_stg,
            rescale_scale=_DEFAULT_VIDEO_RESCALE,
            modality_scale=args.modality_scale,
            skip_step=0,
            stg_blocks=_DEFAULT_STG_BLOCKS,
        ),
        audio_guider_params=MultiModalGuiderParams(
            cfg_scale=args.audio_cfg,
            stg_scale=args.audio_stg,
            rescale_scale=_DEFAULT_AUDIO_RESCALE,
            modality_scale=args.modality_scale,
            skip_step=0,
            stg_blocks=_DEFAULT_STG_BLOCKS,
        ),
        images=images,
        clss_config=clss_config,
        enhance_prompt=args.enhance_prompt,
        max_batch_size=args.batch_size,
        tiling_config=tiling_config,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    logger.info("Decoding and saving video to %s …", args.output)
    encode_video(
        video=video_iter,
        fps=args.fps,
        audio=audio,
        output_path=args.output,
        video_chunks_number=get_video_chunks_number(num_frames, tiling_config),
    )
    logger.info("Done → %s", os.path.abspath(args.output))


if __name__ == "__main__":
    main()
