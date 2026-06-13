"""CLSSStreamingPipeline — streaming text/image-to-video generation using CLSS.

Generates arbitrarily long videos by denoising in temporal chunks and applying
closed-loop corrections between chunks (see clss.py for algorithm details).

Pipeline flow
-------------
1. Encode prompts once (shared across all chunks).
2. Build chunk schedule: list of new_latent_frames per chunk.
3. Keep the transformer loaded across all chunks via DiffusionStage.model_context().
4. For each chunk:
   a. Build CLSS conditionings: overlap (re-noising, §2.1) + anchors (§2.5).
   b. Run DiffusionStage.run() → unpatchified [1, C, total_chunk_lf, H, W].
   c. Extract new frames (drop the conditioned overlap region).
   d. Apply CLSS corrections: AdaIN + frequency shrinkage (§2.3, §2.4).
   e. Update SLB and anchor bank.
   f. Accumulate corrected latent.
5. Concatenate all corrected latents → decode video + audio.

Video/latent frame conversion
------------------------------
The LTX VAE uses a scale factor of 8 along the time axis (causal: the first
pixel frame maps to latent frame 0, then every 8 subsequent pixel frames add
one latent frame):
    latent_frames  = (pixel_frames  − 1) // 8 + 1
    pixel_frames   = (latent_frames − 1) × 8 + 1
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from typing import Optional

import torch

from ltx_core.block_streaming import StreamingModelBuilder
from ltx_core.components.guiders import (
    MultiModalGuiderFactory,
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import AudioPatchifier
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.conditioning import AudioConditionByReferenceLatent
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.primitives import ModelBuilderProtocol
from ltx_core.loader.registry import Registry
from ltx_core.model.transformer.compiling import CompilationConfig
from ltx_core.model.transformer import LTXModel
from ltx_core.model.video_vae.tiling import TilingConfig
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio, AudioLatentShape
from ltx_pipelines.utils import (
    assert_resolution,
    combined_image_conditionings,
    get_device,
)
from ltx_pipelines.utils.args import ImageConditioningInput, detect_checkpoint_path
from ltx_pipelines.utils.blocks import (
    AudioDecoder,
    DiffusionStage,
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
)
from ltx_pipelines.utils.constants import detect_params
from ltx_pipelines.utils.denoisers import FactoryGuidedDenoiser
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.types import ModalitySpec, OffloadMode

from ltx_pipelines.streaming.clss import CLSSConfig, CLSSState

logger = logging.getLogger(__name__)

# VAE temporal down-scale factor: 8 pixel frames → 1 latent frame (causal)
_VAE_TIME_SCALE = 8

# Audio latent temporal rate: 25 latent frames per second
# Derived from: sample_rate=16000 / hop_length=160 / audio_latent_downsample_factor=4
_AUDIO_LATENTS_PER_SEC: float = 16000.0 / 160.0 / 4.0


def _patchify_audio_overlap(
    audio_latent: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Patchify audio overlap latent with positions shifted to just before t=0.

    This mirrors the LipDub audio reference conditioning pattern: the reference
    tokens are given negative RoPE positions so the model interprets them as audio
    that *preceded* the current chunk.  strength=1.0 (clean) is appropriate since
    these are already-denoised latents from the previous chunk.

    Args:
        audio_latent: [B, C, T, mel_bins] — last overlap frames from previous chunk.
        device: target device.

    Returns:
        (patchified [B, T, C*mel_bins], positions [B, 1, T, 2]) with negative time coords.
    """
    patchifier = AudioPatchifier(patch_size=1)
    patchified = patchifier.patchify(audio_latent)
    b, c, _t, mel_bins = audio_latent.shape
    seq_len = patchified.shape[1]
    latent_coords = patchifier.get_patch_grid_bounds(
        output_shape=AudioLatentShape(batch=b, channels=c, frames=seq_len, mel_bins=mel_bins),
        device=device,
    )
    positions = latent_coords.to(dtype=torch.float32)
    # Shift so this audio sits just before t=0 (same convention as LipDub reference audio)
    aud_dur = positions[:, :, -1, 1].max().item()
    positions = positions - aud_dur - 0.04
    return patchified, positions


def _latent_to_pixel_frames(latent_frames: int) -> int:
    return (latent_frames - 1) * _VAE_TIME_SCALE + 1


def _pixel_to_latent_frames(pixel_frames: int) -> int:
    return (pixel_frames - 1) // _VAE_TIME_SCALE + 1


def build_chunk_schedule(total_latent_frames: int, config: CLSSConfig) -> list[int]:
    """Build the list of new latent frames per chunk.

    Each element is the number of *new* latent frames for that chunk (excluding
    the overlap region that is conditioned on the SLB).  The last chunk may be
    smaller than config.new_latent_frames.

    Returns
    -------
    list[int]
        Lengths of consecutive chunks in latent frame count.
        The sum equals total_latent_frames.
    """
    if total_latent_frames <= 0:
        return []
    schedule: list[int] = []
    remaining = total_latent_frames
    while remaining > 0:
        n = min(config.new_latent_frames, remaining)
        schedule.append(n)
        remaining -= n
    return schedule


class CLSSStreamingPipeline:
    """Streaming text/image-to-video pipeline using Closed-Loop Streaming Synthesis.

    Generates long videos (hundreds of frames) by composing many short diffusion
    chunks with CLSS corrections applied between them.  Memory footprint is
    O(overlap_latent_frames + anchor_max_size) latent frames regardless of total
    video length.

    Parameters
    ----------
    checkpoint_path, gemma_root, loras, device, quantization, registry,
    compilation_config, offload_mode:
        Same as TI2VidOneStagePipeline.

    Example
    -------
    ::

        pipeline = CLSSStreamingPipeline(
            checkpoint_path="...",
            gemma_root="...",
            loras=[],
        )
        clss_config = CLSSConfig(tau_c=0.15, beta=0.3, new_latent_frames=21)
        video, audio = pipeline(
            prompt="A river flowing through a forest at sunrise.",
            negative_prompt="",
            seed=42,
            height=480,
            width=704,
            num_frames=257,      # pixel frames
            frame_rate=25.0,
            num_inference_steps=30,
            video_guider_params=MultiModalGuiderParams(cfg_scale=3.0),
            audio_guider_params=MultiModalGuiderParams(cfg_scale=7.0),
            clss_config=clss_config,
        )
    """

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str,
        loras: list[LoraPathStrengthAndSDOps],
        device: Optional[torch.device] = None,
        quantization: Optional[QuantizationPolicy] = None,
        registry: Optional[Registry] = None,
        compilation_config: Optional[CompilationConfig] = None,
        offload_mode: OffloadMode = OffloadMode.NONE,
        # Optional per-component paths (for separate safetensors files)
        embeddings_path: Optional[str] = None,
        video_vae_path: Optional[str] = None,
        audio_vae_path: Optional[str] = None,
        # Optional custom builders (e.g. GGUF-backed builders)
        transformer_builder: Optional[ModelBuilderProtocol[LTXModel]] = None,
        streaming_builder: Optional[StreamingModelBuilder] = None,
        streaming_text_encoder_builder=None,
    ) -> None:
        self.dtype = torch.bfloat16
        self.device = device or get_device()
        self._scheduler = LTX2Scheduler()
        self.prompt_encoder = PromptEncoder(
            checkpoint_path=embeddings_path or checkpoint_path,
            gemma_root=gemma_root,
            dtype=self.dtype,
            device=self.device,
            registry=registry,
            offload_mode=offload_mode,
            streaming_text_encoder_builder=streaming_text_encoder_builder,
        )
        self.image_conditioner = ImageConditioner(
            checkpoint_path=video_vae_path or checkpoint_path,
            dtype=self.dtype,
            device=self.device,
            registry=registry,
        )
        self.stage = DiffusionStage(
            checkpoint_path=checkpoint_path,
            dtype=self.dtype,
            device=self.device,
            loras=tuple(loras),
            quantization=quantization,
            registry=registry,
            compilation_config=compilation_config,
            offload_mode=offload_mode,
            transformer_builder=transformer_builder,
            streaming_builder=streaming_builder,
        )
        self.video_decoder = VideoDecoder(
            checkpoint_path=video_vae_path or checkpoint_path,
            dtype=self.dtype,
            device=self.device,
            registry=registry,
        )
        self.audio_decoder = AudioDecoder(
            checkpoint_path=audio_vae_path or checkpoint_path,
            dtype=self.dtype,
            device=self.device,
            registry=registry,
        )

    def __call__(  # noqa: PLR0913
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        video_guider_params: MultiModalGuiderParams | MultiModalGuiderFactory,
        audio_guider_params: MultiModalGuiderParams | MultiModalGuiderFactory,
        images: list[ImageConditioningInput] | None = None,
        clss_config: Optional[CLSSConfig] = None,
        enhance_prompt: bool = False,
        tiling_config: Optional[TilingConfig] = None,
        max_batch_size: int = 1,
        sigmas: Optional[torch.Tensor] = None,
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        """Generate *num_frames* pixel frames as a streaming CLSS video.

        Parameters
        ----------
        prompt, negative_prompt, seed, height, width, frame_rate,
        num_inference_steps, video_guider_params, audio_guider_params,
        images, enhance_prompt, tiling_config, max_batch_size, sigmas:
            Same semantics as TI2VidOneStagePipeline.
        num_frames:
            Total number of pixel frames to generate.  Will be rounded down to
            the nearest value satisfying the latent frame count constraint:
            actual = (latent_frames − 1) × 8 + 1.
        clss_config:
            CLSS algorithm configuration.  Defaults to CLSSConfig() with the
            paper-recommended values (tau_c=0.15, beta=0.3, new_latent_frames=21).
        """
        if clss_config is None:
            clss_config = CLSSConfig()
        if images is None:
            images = []

        assert_resolution(height=height, width=width, is_two_stage=False)

        # Log stability estimate — ρ_loop = (1−β)·(1−τc), target < 1.0 (§2.6)
        rho = (1.0 - clss_config.beta) * (1.0 - clss_config.tau_c)
        logger.info("CLSS loop gain estimate ρ_loop ≈ %.3f (stability margin: target < 1.0)", rho)
        if rho >= 1.0:
            logger.warning(
                "ρ_loop = %.3f ≥ 1.0 — the loop may be unstable.  "
                "Increase beta or tau_c (§2.6).",
                rho,
            )

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)

        # ------------------------------------------------------------------
        # Encode prompts (once, shared across all chunks)
        # ------------------------------------------------------------------
        ctx_p, ctx_n = self.prompt_encoder(
            [prompt, negative_prompt],
            enhance_first_prompt=enhance_prompt,
            enhance_prompt_seed=seed,
        )
        v_ctx_p, a_ctx_p = ctx_p.video_encoding, ctx_p.audio_encoding
        v_ctx_n, a_ctx_n = ctx_n.video_encoding, ctx_n.audio_encoding

        # Prompt diagnostics — detect degenerate context (weak CFG = prompt has no effect).
        # NOTE: do NOT use full-tensor cosine similarity here.  Both sequences are padded to
        # the same max length (1024 tokens).  Positions beyond the actual prompt length contain
        # the same padding-token embedding in both cond and uncond, which dominates the dot
        # product and makes the cosine artificially high (typically ~0.97).  The padding
        # cancels exactly in cond-uncond, so CFG is not affected.  Use
        # ||cond - uncond|| / ||cond|| as the guidance-strength metric instead.
        with torch.no_grad():
            v_p_norm = v_ctx_p.float().norm().item()
            v_n_norm = v_ctx_n.float().norm().item()
            v_diff_norm = (v_ctx_p - v_ctx_n).float().norm().item()
            v_cfg_strength = v_diff_norm / (v_p_norm + 1e-8)
        logger.info(
            "[prompt] video ctx  pos_norm=%.1f  neg_norm=%.1f  "
            "||cond-uncond||/||cond||=%.4f  "
            "(< 0.05 → CFG direction is tiny, prompt has little effect; "
            "padding inflates cosine—use this ratio instead)",
            v_p_norm, v_n_norm, v_cfg_strength,
        )
        if v_cfg_strength < 0.05:
            logger.warning(
                "[prompt] CFG direction is very weak (%.4f < 0.05).  "
                "Try a stronger negative prompt or check embeddings processor output.",
                v_cfg_strength,
            )
        if v_ctx_p is None:
            logger.error("[prompt] v_ctx_p is None — video context is missing, generation will be unconditioned!")
        if v_ctx_n is None:
            logger.error("[prompt] v_ctx_n is None — negative context missing, CFG will be disabled!")
        if a_ctx_p is None:
            logger.warning("[prompt] a_ctx_p is None — audio context missing, audio generation is unconditioned")

        video_guider_factory = create_multimodal_guider_factory(
            params=video_guider_params,
            negative_context=v_ctx_n,
        )
        audio_guider_factory = create_multimodal_guider_factory(
            params=audio_guider_params,
            negative_context=a_ctx_n,
        )

        # ------------------------------------------------------------------
        # Encode first-frame image conditionings (for chunk 0 only)
        # ------------------------------------------------------------------
        first_chunk_conditionings = self.image_conditioner(
            lambda enc: combined_image_conditionings(
                images=images,
                height=height,
                width=width,
                video_encoder=enc,
                dtype=self.dtype,
                device=self.device,
            )
        ) if images else []

        # ------------------------------------------------------------------
        # Sigma schedule (same for every chunk)
        # ------------------------------------------------------------------
        chunk_sigmas = (
            sigmas if sigmas is not None
            else self._scheduler.execute(steps=num_inference_steps)
        ).to(dtype=torch.float32, device=self.device)

        # ------------------------------------------------------------------
        # Chunk schedule
        # ------------------------------------------------------------------
        total_latent_frames = _pixel_to_latent_frames(num_frames)
        chunk_schedule = build_chunk_schedule(total_latent_frames, clss_config)

        if not chunk_schedule:
            raise ValueError(f"num_frames={num_frames} produced an empty chunk schedule.")

        logger.info(
            "CLSS streaming: %d pixel frames → %d latent frames → %d chunks "
            "(overlap=%d latent, new=%d latent per chunk).  "
            "ρ_loop estimate: %.3f",
            num_frames,
            total_latent_frames,
            len(chunk_schedule),
            clss_config.overlap_latent_frames,
            clss_config.new_latent_frames,
            rho,
        )

        # ------------------------------------------------------------------
        # CLSS state
        # ------------------------------------------------------------------
        clss_state = CLSSState(clss_config)

        all_new_video_latents: list[torch.Tensor] = []
        all_new_audio_latents: list[torch.Tensor] = []

        # Audio overlap conditioning: last overlap_audio_lf frames of the previous chunk's audio.
        # Passed as negative-position reference tokens to help the model continue audio smoothly.
        prev_audio_overlap: Optional[torch.Tensor] = None
        # Precompute how many audio latent frames correspond to one video overlap period (constant).
        _overlap_audio_lf = round(
            _latent_to_pixel_frames(clss_config.overlap_latent_frames) / frame_rate * _AUDIO_LATENTS_PER_SEC
        )

        # ------------------------------------------------------------------
        # Keep the transformer in GPU memory across all chunks
        # ------------------------------------------------------------------
        with self.stage.model_context() as transformer:
            for chunk_idx, chunk_new_lf in enumerate(chunk_schedule):
                is_first = chunk_idx == 0
                overlap_lf = 0 if is_first else clss_config.overlap_latent_frames
                total_chunk_lf = overlap_lf + chunk_new_lf
                chunk_pixel_frames = _latent_to_pixel_frames(total_chunk_lf)

                # Build CLSS conditionings ---------------------------------
                #   §2.1 re-noising: overlap via VideoConditionByLatentIndex
                #   §2.5 anchors:   keyframes via VideoConditionByKeyframeIndex
                overlap_conds = clss_state.get_overlap_conditioning()
                anchor_conds = clss_state.get_anchor_conditioning()
                user_conds = first_chunk_conditionings if is_first else []
                all_conditionings = overlap_conds + anchor_conds + user_conds

                logger.info(
                    "[CLSS] chunk=%d/%d  latent_frames=%d (overlap=%d + new=%d)  "
                    "pixel_frames=%d  conditionings: %d overlap, %d anchors",
                    chunk_idx + 1,
                    len(chunk_schedule),
                    total_chunk_lf,
                    overlap_lf,
                    chunk_new_lf,
                    chunk_pixel_frames,
                    len(overlap_conds),
                    len(anchor_conds),
                )

                # Audio overlap conditioning: reference tokens from the previous chunk's tail
                # positioned "before" t=0 so the model continues from where audio left off.
                audio_conds = []
                if prev_audio_overlap is not None:
                    ref_patch, ref_pos = _patchify_audio_overlap(prev_audio_overlap, self.device)
                    audio_conds = [AudioConditionByReferenceLatent(ref_patch, ref_pos, strength=1.0)]
                    logger.debug(
                        "[CLSS] chunk=%d/%d  audio reference conditioning: %d audio latent frames",
                        chunk_idx + 1, len(chunk_schedule), prev_audio_overlap.shape[2],
                    )

                denoiser = FactoryGuidedDenoiser(
                    v_context=v_ctx_p,
                    a_context=a_ctx_p,
                    video_guider_factory=video_guider_factory,
                    audio_guider_factory=audio_guider_factory,
                )

                # Denoising loop for this chunk ----------------------------
                t_denoise_start = time.perf_counter()
                video_state, audio_state = self.stage.run(
                    transformer=transformer,
                    denoiser=denoiser,
                    sigmas=chunk_sigmas,
                    noiser=noiser,
                    width=width,
                    height=height,
                    frames=chunk_pixel_frames,
                    fps=frame_rate,
                    video=ModalitySpec(
                        context=v_ctx_p,
                        conditionings=all_conditionings,
                    ),
                    audio=ModalitySpec(context=a_ctx_p, conditionings=audio_conds),
                    max_batch_size=max_batch_size,
                )
                t_denoise = time.perf_counter() - t_denoise_start

                # video_state.latent: [1, C, total_chunk_lf, H, W] (unpatchified)
                chunk_video_latent = video_state.latent

                # Drop the overlap region — it was already output in the previous chunk
                new_video_latent = chunk_video_latent[:, :, overlap_lf:]  # [1, C, new_lf, H, W]

                # §2.3 AdaIN + §2.4 frequency shrinkage (updates EMA refs internally)
                t_post_start = time.perf_counter()
                corrected_video = clss_state.post_process(new_video_latent)
                t_post = time.perf_counter() - t_post_start

                # Boundary continuity: compare last frame of previous chunk to first of current
                if all_new_video_latents:
                    with torch.no_grad():
                        prev_last = all_new_video_latents[-1][:, :, -1:].float()
                        curr_first = corrected_video[:, :, :1].float()
                        boundary_l2 = (prev_last - curr_first).norm().item()
                        p_feat = torch.nn.functional.normalize(prev_last.flatten(1), dim=1)
                        c_feat = torch.nn.functional.normalize(curr_first.flatten(1), dim=1)
                        boundary_sim = (p_feat * c_feat).sum().item()
                    logger.info(
                        "[CLSS] chunk=%d/%d  boundary  cosine_sim=%.4f  l2_dist=%.4f",
                        chunk_idx + 1, len(chunk_schedule), boundary_sim, boundary_l2,
                    )

                # Intra-chunk drift: how much did the scene change within this chunk?
                if logger.isEnabledFor(logging.DEBUG) and corrected_video.shape[2] > 1:
                    with torch.no_grad():
                        c_first = torch.nn.functional.normalize(corrected_video[:, :, :1].float().flatten(1), dim=1)
                        c_last  = torch.nn.functional.normalize(corrected_video[:, :, -1:].float().flatten(1), dim=1)
                        intra_sim = (c_first * c_last).sum().item()
                    logger.debug(
                        "[CLSS] chunk=%d/%d  intra_chunk_sim=%.4f  (cosine first→last new frame; "
                        "low = content drifted within this chunk)",
                        chunk_idx + 1, len(chunk_schedule), intra_sim,
                    )

                # Update SLB and anchor bank (§2.5)
                clss_state.update_buffer(corrected_video)

                logger.info(
                    "[CLSS] chunk=%d/%d  timing  denoise=%.1fs  post_process=%.3fs",
                    chunk_idx + 1, len(chunk_schedule), t_denoise, t_post,
                )

                all_new_video_latents.append(corrected_video)

                # Audio: drop overlap frames from non-first chunks so audio duration
                # matches video duration.  The overlap period was already covered by
                # the previous chunk's audio — including it again would desynchronise
                # the audio/video timeline.
                if audio_state is not None:
                    audio_latent = audio_state.latent  # [B, C, A_lf, mel_bins]
                    if not is_first:
                        overlap_px = _latent_to_pixel_frames(overlap_lf)
                        overlap_audio_lf = round(overlap_px / frame_rate * _AUDIO_LATENTS_PER_SEC)
                        audio_latent = audio_latent[:, :, overlap_audio_lf:]
                        logger.debug(
                            "[CLSS] chunk=%d/%d  audio overlap drop: %d audio latent frames "
                            "(video overlap: %d lf = %d px = %.3f s)",
                            chunk_idx + 1, len(chunk_schedule),
                            overlap_audio_lf, overlap_lf, overlap_px, overlap_px / frame_rate,
                        )
                    all_new_audio_latents.append(audio_latent)

                    # Save the tail of this chunk's new audio for next chunk's reference conditioning.
                    # Use the minimum of _overlap_audio_lf and available frames to handle short chunks.
                    ref_len = min(_overlap_audio_lf, audio_latent.shape[2])
                    if ref_len > 0:
                        prev_audio_overlap = audio_latent[:, :, -ref_len:].clone()

        # ------------------------------------------------------------------
        # Concatenate all chunks and decode
        # ------------------------------------------------------------------
        full_video_latent = torch.cat(all_new_video_latents, dim=2)  # [1, C, total_lf, H, W]
        logger.info(
            "CLSS done: %d chunks, full latent shape %s",
            len(chunk_schedule),
            tuple(full_video_latent.shape),
        )

        # Free everything that is no longer needed before VAE decode.
        # The transformer is already on meta (freed by model_context()), but the
        # prompt context tensors, per-chunk latent list, guider factories, and audio
        # overlap buffer are still allocated.  Releasing them recovers several GB of
        # VRAM that the VAE decoder needs for long videos.
        del all_new_video_latents       # chunks are now in full_video_latent
        del v_ctx_p, v_ctx_n, a_ctx_p, a_ctx_n
        del video_guider_factory, audio_guider_factory
        del prev_audio_overlap
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("GPU memory freed before VAE decode.")

        # Audio is decoded eagerly (small latent, fast); video is decoded lazily
        # via the tiled iterator so only one tile is on GPU at a time.
        if all_new_audio_latents:
            full_audio_latent = torch.cat(all_new_audio_latents, dim=2)
            del all_new_audio_latents
            decoded_audio = self.audio_decoder(full_audio_latent)
            del full_audio_latent
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            decoded_audio = Audio(waveform=torch.zeros(1, 0), sampling_rate=16000)

        decoded_video = self.video_decoder(full_video_latent, tiling_config, generator=generator)

        return decoded_video, decoded_audio


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


@torch.inference_mode()
def main() -> None:
    """Command-line entry point for CLSSStreamingPipeline."""
    import argparse
    logging.basicConfig(level=logging.INFO)
    checkpoint_path = detect_checkpoint_path()
    params = detect_params(checkpoint_path)

    # Reuse the standard 1-stage argument parser and add CLSS-specific args
    from ltx_pipelines.utils.args import default_1_stage_arg_parser

    parser = default_1_stage_arg_parser(params=params)

    g = parser.add_argument_group("CLSS streaming")
    g.add_argument("--clss-tau-c", type=float, default=0.15,
                   help="Context re-noising level τc (§2.1).  Range [0, 0.2].")
    g.add_argument("--clss-beta", type=float, default=0.3,
                   help="AdaIN correction blend factor β (§2.3).  Range [0, 0.5].")
    g.add_argument("--clss-ema-lambda", type=float, default=0.05,
                   help="EMA update rate per chunk (§2.3/§2.4).")
    g.add_argument("--clss-overlap", type=int, default=4,
                   help="Overlap latent frames (SLB size).")
    g.add_argument("--clss-new-frames", type=int, default=21,
                   help="New latent frames per chunk.")

    args = parser.parse_args()

    clss_config = CLSSConfig(
        tau_c=args.clss_tau_c,
        beta=args.clss_beta,
        ema_lambda=args.clss_ema_lambda,
        overlap_latent_frames=args.clss_overlap,
        new_latent_frames=args.clss_new_frames,
    )

    pipeline = CLSSStreamingPipeline(
        checkpoint_path=args.checkpoint_path,
        gemma_root=args.gemma_root,
        loras=tuple(args.lora) if args.lora else (),
        quantization=args.quantization,
        compilation_config=args.compile,
        offload_mode=args.offload_mode,
    )

    video, audio = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        num_inference_steps=args.num_inference_steps,
        video_guider_params=MultiModalGuiderParams(
            cfg_scale=args.video_cfg_guidance_scale,
            stg_scale=args.video_stg_guidance_scale,
            rescale_scale=args.video_rescale_scale,
            modality_scale=args.a2v_guidance_scale,
            skip_step=args.video_skip_step,
            stg_blocks=args.video_stg_blocks,
        ),
        audio_guider_params=MultiModalGuiderParams(
            cfg_scale=args.audio_cfg_guidance_scale,
            stg_scale=args.audio_stg_guidance_scale,
            rescale_scale=args.audio_rescale_scale,
            modality_scale=args.v2a_guidance_scale,
            skip_step=args.audio_skip_step,
            stg_blocks=args.audio_stg_blocks,
        ),
        images=args.images,
        clss_config=clss_config,
    )

    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        output_path=args.output_path,
        video_chunks_number=1,
    )


if __name__ == "__main__":
    main()
