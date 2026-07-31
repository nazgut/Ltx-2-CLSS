"""
Closed-Loop Streaming Synthesis (CLSS) algorithm components.

CLSS extends Latent Streaming Synthesis (LSS) with four closed-loop corrections that
eliminate the exposure-bias drift accumulation inherent in open-loop streaming:

  §2.1  Calibrated context re-noising
        The overlap latents fed as context to chunk N are re-noised to level τc:
            L̃_overlap = α(τc)·L̂_overlap + σ(τc)·ε   (α = 1−τc, σ = τc)
        In the LTX framework this is achieved by conditioning with
        VideoConditionByLatentIndex(strength=1−τc).  The existing GaussianNoiser
        then naturally applies the formula; the denoising transformer sees those
        tokens tagged with timestep σ·τc instead of 0, so it can actively
        re-project the re-noised context back onto the data manifold.

  §2.3  EMA-tracked per-channel distribution reference
        A slow EMA (rate λ ≈ 0.05) tracks per-channel mean and std across chunks.
        Fast drift (single-chunk statistical error) is suppressed; slow intended
        evolution (lighting changes, scene content) passes through.  Applied as a
        per-channel AdaIN-style renormalisation blended with factor β.

  §2.4  Frequency-band soft shrinkage
        The latent is decomposed into B spatio-temporal frequency bands via a 3-D
        FFT.  Each band b receives a continuous gain min(1, (E_ref_b / E_b)^γ_b).
        Only bands with excess energy relative to their EMA reference are
        attenuated, and only by the excess, so on-reference texture passes
        untouched while temporal flicker (high temporal frequency bands) is
        suppressed.

  §2.5  Dynamic anchor bank
        A small bank of anchor keyframes is maintained.  The first frame of the
        first chunk seeds the bank.  Persistent cosine-similarity drops (below
        threshold ρ for two consecutive chunks) trigger a new anchor.  Chunk N
        cross-attends to the top-m most similar anchors, providing soft long-range
        identity without dragging back to a single frozen frame-0 reference.
        Because CLIP is not bundled with LTX, anchor similarity is computed from
        mean-pooled, L2-normalised latent features.

Algorithm 1 (per-chunk step):
  1.  L_overlap  ← SLB.read()
  2.  L̃_overlap ← α(τc)·L_overlap + σ(τc)·ε          [§2.1, via conditioning]
  3.  A_m        ← top-m anchors by feature similarity  [§2.5]
  4.  L_N        ← Generate(L̃_overlap, A_m, prompt)
  5.  L_N        ← AdaIN lerp toward per-channel EMA ref, factor β  [§2.3]
  6.  L_N        ← band-wise soft shrinkage              [§2.4]
  7.  EMA refs   ← update(L_N); anchor bank ← update on persistent scene change
  8.  SLB.push(trailing frames)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

from ltx_core.conditioning import VideoConditionByKeyframeIndex, VideoConditionByLatentIndex
from ltx_core.conditioning.item import ConditioningItem

logger = logging.getLogger(__name__)


def _chan_stats(x: torch.Tensor) -> tuple[list[float], list[float]]:
    """Return (per-channel means, per-channel stds) for a [B, C, F, H, W] latent."""
    with torch.no_grad():
        flat = x.float().permute(1, 0, 2, 3, 4).flatten(1)
        return flat.mean(1).tolist(), flat.std(1).tolist()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CLSSConfig:
    """Hyperparameters for Closed-Loop Streaming Synthesis.

    Parameters
    ----------
    tau_c:
        Re-noising noise level applied to overlap context (§2.1).
        0 recovers LSS behaviour (maximal continuity, maximal drift accumulation);
        larger values give more distributional repair at the cost of softer
        motion lock.  Recommended range: 0.10–0.20.
    beta:
        AdaIN correction blend factor (§2.3).  0 = no correction, 1 = full
        replacement with reference statistics.  Recommended: 0.2–0.4.
    ema_lambda:
        EMA update rate per chunk (§2.3 / §2.4).  Slow enough to track intended
        scene evolution while suppressing per-chunk drift.  Recommended: 0.05–0.10.
    n_freq_bands:
        Number of 3-D FFT frequency bands (§2.4).  4 is sufficient for most cases.
    freq_gamma:
        Per-band shrinkage exponent γ_b (§2.4, Eq. 5).  Element 0 = DC band
        (should be 0.0, never shrink structure/motion); last element = highest
        temporal-frequency band (flicker).  Length must equal n_freq_bands.
    anchor_threshold:
        Cosine-similarity threshold below which a chunk is considered a candidate
        scene change (§2.5).  Default 0.78 (from the paper).
    anchor_persistence:
        Number of consecutive below-threshold chunks required before a new anchor
        is committed (persistence test, §2.5).  Filters transient occlusions.
    anchor_max_size:
        Maximum number of anchors retained in the bank (§2.5).  Oldest
        least-recently-retrieved anchor is evicted when the cap is exceeded.
    anchor_top_m:
        Number of anchors retrieved per chunk for conditioning (§2.5).
    anchor_strength:
        Denoising strength for anchor keyframe tokens (1.0 = fully clean / no
        denoising; use 1.0 to prevent anchor frames from being corrupted).
    overlap_latent_frames:
        Number of latent frames shared between consecutive chunks (the SLB size).
        Corresponds to the temporal context the model sees from the previous chunk.
    new_latent_frames:
        Number of genuinely new latent frames generated per chunk.  The pixel
        frame count is (new_latent_frames − 1) × 8 + 1 with the default VAE
        scale factor of 8.
    """

    # §2.1  Calibrated context re-noising
    # 0.0 = full overlap strength (maximal continuity, some drift risk)
    # 0.05 = very light re-noising, very strong temporal constraint
    # 0.15–0.20 = paper recommended range (too loose for 22B GGUF streaming)
    tau_c: float = 0.05

    # §2.3  EMA-tracked per-channel distribution reference
    beta: float = 0.4
    ema_lambda: float = 0.05
    # Cap on how far the per-channel EMA std may grow relative to its initial
    # value (chunk-0 statistics).  0.0 = uncapped (old behaviour, allows σ drift).
    # With the default 0.05 the reference std is allowed to increase at most 5 %
    # from chunk 0's value, which prevents AdaIN from quietly amplifying late chunks
    # while still permitting slow intentional brightening / saturation changes.
    ema_sigma_max_drift: float = 0.05
    # Maximum per-channel AdaIN upward amplification factor.
    # When the EMA reference std for a channel exceeds the current chunk's std,
    # AdaIN would amplify that channel's variance — boosting any residual
    # denoising noise and causing visible grain in the decoded video.
    # This cap limits how aggressively AdaIN can amplify: 1.2 = allow at most
    # 20 % upward scaling.  0.0 = no cap (original behaviour).
    # Recommended: 1.2 when noise/grain is visible, especially with < 30 steps.
    adain_max_amplification: float = 0.0

    # §2.4  Frequency-band soft shrinkage
    # Keep gamma conservative: high values (0.5+) over-attenuate mid-frequency
    # content and degrade cosine similarity at chunk boundaries.
    n_freq_bands: int = 4
    # Band 0 = DC (never shrink: would kill global color/brightness).
    # Band 1 = low spatial/temporal frequency: small γ prevents gradual energy
    #          compounding without touching scene structure (observed +18-19 %
    #          uncorrected growth over 17 chunks in testing).
    freq_gamma: tuple[float, ...] = (0.0, 0.05, 0.2, 0.3)

    # §2.5  Dynamic anchor bank
    anchor_threshold: float = 0.78
    anchor_persistence: int = 2
    anchor_max_size: int = 8
    anchor_top_m: int = 2
    anchor_strength: float = 1.0
    # Force a new anchor every N chunks regardless of similarity.  0 = disabled.
    # Without this, on visually-stable scenes the bank stagnates (similarity stays
    # above threshold, persistence never fires) and the model loses recent reference
    # frames — the primary cause of content collapse after ~10-15 chunks.
    anchor_force_every: int = 5

    # Streaming buffer dimensions
    # 8 latent frames ≈ 57 pixel frames ≈ 2.4 s of hard context from previous chunk.
    # Combined with tau_c=0.05 this makes the overlap frames nearly frozen, giving
    # strong visual continuity at the cost of ~17% more denoising work per chunk.
    overlap_latent_frames: int = 8
    new_latent_frames: int = 13

    # §3.7  Open-loop model gain measurement.
    # measure_g: run a second denoising pass with a perturbed overlap on every chunk
    #   that has overlap, to track g = ||Δ_output||_F / ||δ_input||_F per-chunk.
    #   Doubles denoising time for non-first chunks.  Requires a non-None generator.
    # measure_g_epsilon: perturbation magnitude as a fraction of the overlap norm.
    measure_g: bool = True
    measure_g_epsilon: float = 0.01

    # Temporally-correlated initial video noise (EXPERIMENTAL, off by default —
    # unvalidated until a live run).  Targets the measured ~4 s layout
    # oscillation, which is invariant to chunk length / overlap length / SLB
    # strength: with i.i.d. noise every ~4 s span of frames carries an
    # independent low-frequency "content suggestion" that the model resolves as
    # a fresh motion arc at its trained temporal horizon.  Mixing a run-constant
    # shared frame into every video noise frame,
    #     n_t = sqrt(1-a)·eps_t + sqrt(a)·eps_shared,
    # keeps each frame's marginal exactly N(0,1) but raises frame-to-frame noise
    # correlation to a at ALL lags (inference-time correlated noise prior, same
    # family as FreeNoise noise rescheduling / PYoCo mixed noise).  Unlike
    # block-repeat schemes this cannot introduce a new periodicity.  0.0 = off
    # (bit-exact baseline).  Worst case behaves like a seed change with more
    # static content.
    noise_temporal_corr: float = 0.0

    def __post_init__(self) -> None:
        if len(self.freq_gamma) != self.n_freq_bands:
            raise ValueError(
                f"len(freq_gamma)={len(self.freq_gamma)} must equal n_freq_bands={self.n_freq_bands}"
            )
        if not 0.0 <= self.tau_c <= 1.0:
            raise ValueError(f"tau_c must be in [0, 1], got {self.tau_c}")
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError(f"beta must be in [0, 1], got {self.beta}")
        if not 0.0 <= self.noise_temporal_corr < 1.0:
            raise ValueError(
                f"noise_temporal_corr must be in [0, 1), got {self.noise_temporal_corr}"
            )


# ---------------------------------------------------------------------------
# §2.3  Per-channel EMA reference
# ---------------------------------------------------------------------------


class _PerChannelEMA:
    """Exponential moving average of per-channel mean and standard deviation.

    Operates on unpatchified latents of shape [B, C, F, H, W].
    Statistics are computed over all non-channel dimensions (B, F, H, W).
    """

    def __init__(self) -> None:
        self.mean: Optional[torch.Tensor] = None  # [C]
        self.std: Optional[torch.Tensor] = None   # [C]
        self._init_std: Optional[torch.Tensor] = None  # [C] anchored to chunk-0

    def update(self, latent: torch.Tensor, lam: float, sigma_max_drift: float = 0.0) -> None:
        """Update EMA statistics.  latent: [B, C, F, H, W]

        sigma_max_drift: if > 0, caps the per-channel EMA std to at most
        (1 + sigma_max_drift) × the chunk-0 std, preventing gradual amplification.
        """
        # [C, B*F*H*W]
        x = latent.float().permute(1, 0, 2, 3, 4).flatten(1)
        mu = x.mean(1)
        sig = x.std(1).clamp(min=1e-5)
        if self.mean is None:
            self.mean = mu.clone()
            self.std = sig.clone()
            self._init_std = sig.clone()
        else:
            self.mean = (1.0 - lam) * self.mean + lam * mu
            self.std  = (1.0 - lam) * self.std  + lam * sig
            if sigma_max_drift > 0.0 and self._init_std is not None:
                self.std = self.std.clamp(max=self._init_std * (1.0 + sigma_max_drift))

    def apply_adain(
        self, latent: torch.Tensor, beta: float, max_amplification: float = 0.0
    ) -> torch.Tensor:
        """Blend *latent* toward the EMA statistics via per-channel AdaIN.

        Result = (1−β)·latent + β·AdaIN(latent → EMA).
        Returns the original latent unchanged when the EMA is uninitialised
        (first chunk).

        max_amplification: if > 0, caps per-channel upward std scaling to this
        factor.  E.g. 1.2 = allow the EMA reference to push a channel's std up
        by at most 20 %.  Attenuation (EMA std < current std) is never capped.
        Set to 0.0 to disable (original behaviour).
        """
        if self.mean is None:
            return latent
        B, C, F, H, W = latent.shape
        # [C, N] where N = B*F*H*W
        x = latent.float().permute(1, 0, 2, 3, 4).flatten(1)
        mu_cur = x.mean(1, keepdim=True)
        sig_cur = x.std(1, keepdim=True).clamp(min=1e-5)
        # Per-channel target std: optionally cap upward amplification so that
        # channels with EMA std >> current std don't boost residual denoising noise.
        target_std = self.std.unsqueeze(1)  # [C, 1]
        if max_amplification > 0.0:
            cap = sig_cur * max_amplification  # [C, 1]
            target_std = torch.minimum(target_std, cap)
        # Normalise to zero-mean/unit-std, then scale to (capped) EMA reference
        corrected = (x - mu_cur) / sig_cur * target_std + self.mean.unsqueeze(1)
        blended = (1.0 - beta) * x + beta * corrected
        return blended.view(C, B, F, H, W).permute(1, 0, 2, 3, 4).to(latent.dtype)


# ---------------------------------------------------------------------------
# §2.4  Frequency-band soft shrinkage
# ---------------------------------------------------------------------------


@dataclass
class _BandEMARef:
    """Per-band EMA of energy (mean squared amplitude) per channel."""
    energy: list[torch.Tensor] = field(default_factory=list)  # each [C]

    def update(self, energies: list[torch.Tensor], lam: float) -> None:
        if not self.energy:
            self.energy = [e.clone() for e in energies]
        else:
            self.energy = [
                (1.0 - lam) * ref + lam * e
                for ref, e in zip(self.energy, energies)
            ]


def _compute_radial_freq(
    F_size: int, H_size: int, W_size: int, device: torch.device
) -> torch.Tensor:
    """Return [F, H, W] tensor of max-axis normalised frequency in [0, 0.5]."""
    ff = torch.fft.fftfreq(F_size, device=device).abs().view(F_size, 1, 1)
    fh = torch.fft.fftfreq(H_size, device=device).abs().view(1, H_size, 1)
    fw = torch.fft.fftfreq(W_size, device=device).abs().view(1, 1, W_size)
    # Use the maximum frequency across all three dimensions so that a
    # spatial-only high-frequency pattern is treated the same as a temporal one.
    return torch.maximum(torch.maximum(ff.expand(F_size, H_size, W_size),
                                        fh.expand(F_size, H_size, W_size)),
                          fw.expand(F_size, H_size, W_size))


def _apply_band_soft_shrinkage(
    latent: torch.Tensor,
    band_ema: _BandEMARef,
    n_bands: int,
    gamma: tuple[float, ...],
    lam: float,
    chunk_idx: int = -1,
) -> torch.Tensor:
    """Decompose *latent* into frequency bands and apply per-band soft shrinkage.

    Only bands whose current energy exceeds the EMA reference are attenuated
    (gain ≤ 1).  Bands with γ = 0 are never attenuated (DC and low-frequency
    structure is preserved exactly).  EMA references are updated after gains
    are applied so that the reference tracks the corrected, not the raw, energy.

    latent: [B, C, F, H, W]
    Returns corrected latent with the same shape and dtype.
    """
    B, C, F, H, W = latent.shape
    L = torch.fft.fftn(latent.float(), dim=(-3, -2, -1))  # [B, C, F, H, W] complex

    radial = _compute_radial_freq(F, H, W, latent.device)  # [F, H, W]
    thresholds = torch.linspace(0.0, 0.5, n_bands + 1, device=latent.device)

    band_ffts: list[torch.Tensor] = []
    raw_energies: list[torch.Tensor] = []  # before shrinkage, [C]

    for i in range(n_bands):
        lo = thresholds[i]
        hi = thresholds[i + 1] if i < n_bands - 1 else thresholds[i + 1] + 1.0
        mask = ((radial >= lo) & (radial < hi)).float()  # [F, H, W]
        band_fft = L * mask.view(1, 1, F, H, W)
        band_ffts.append(band_fft)
        # Mean squared amplitude per channel: E_b^(c) = mean |L_b^(c)|^2
        energy = band_fft.abs().pow(2).mean(dim=(0, 2, 3, 4)).clamp(min=1e-10)  # [C]
        raw_energies.append(energy)

    result_fft = torch.zeros_like(L)
    corrected_energies: list[torch.Tensor] = []
    _band_gains: list[float] = []  # mean gain per band for debug logging

    for i, (bfft, g) in enumerate(zip(band_ffts, gamma)):
        if g == 0.0 or not band_ema.energy:
            # No shrinkage; pass through unchanged
            result_fft = result_fft + bfft
            corrected_energies.append(raw_energies[i])
            _band_gains.append(1.0)
        else:
            e_ref = band_ema.energy[i]          # [C] EMA reference
            e_cur = raw_energies[i]             # [C] current energy
            # gain_b^(c) = min(1, (E_ref_b^(c) / E_cur_b^(c))^γ_b) ∈ (0, 1]
            gain = (e_ref / e_cur).pow(g).clamp(max=1.0)  # [C]
            gain_bcast = gain.view(1, C, 1, 1, 1)
            shrunk = bfft * gain_bcast
            result_fft = result_fft + shrunk
            # Post-shrinkage energy: gain^2 * E_cur (in expectation)
            corrected_energies.append(gain.pow(2) * e_cur)
            _band_gains.append(gain.mean().item())

    if logger.isEnabledFor(logging.DEBUG):
        for i in range(n_bands):
            e_raw_mean = raw_energies[i].mean().item()
            e_cor_mean = corrected_energies[i].mean().item()
            e_ref_mean = band_ema.energy[i].mean().item() if band_ema.energy else float("nan")
            logger.debug(
                "[CLSS] chunk=%d  freq_band=%d  γ=%.1f  "
                "E_raw=%.4e  E_ref=%.4e  E_cor=%.4e  gain_mean=%.4f",
                chunk_idx, i, gamma[i],
                e_raw_mean, e_ref_mean, e_cor_mean, _band_gains[i],
            )

    # §3.5 always-on telemetry: band gains and energies (backs paper claim "gains in [0.982,0.999]")
    _e_ref_means = [
        band_ema.energy[i].mean().item() if band_ema.energy else float("nan")
        for i in range(n_bands)
    ]
    print(
        f"[CLSS] chunk={chunk_idx}"
        f"  band_gain=[{', '.join(f'{g:.4f}' for g in _band_gains)}]"
        f"  band_E=[{', '.join(f'{raw_energies[i].mean().item():.3e}' for i in range(n_bands))}]"
        f"  band_Eref=[{', '.join(f'{_e_ref_means[i]:.3e}' for i in range(n_bands))}]"
    )

    # Update EMA with the corrected (post-gain) energies
    band_ema.update(corrected_energies, lam)

    out = torch.fft.ifftn(result_fft, dim=(-3, -2, -1)).real
    return out.to(latent.dtype)


# ---------------------------------------------------------------------------
# §2.5  Dynamic anchor bank
# ---------------------------------------------------------------------------


@dataclass
class _AnchorEntry:
    """Single keyframe anchor."""
    feature: torch.Tensor    # [C] L2-normalised mean-pooled latent feature
    latent: torch.Tensor     # [1, C, 1, H, W] single-frame latent
    frame_idx: int           # absolute latent frame index in the full generated video
    last_retrieved: int = 0  # chunk index of most recent retrieval (for LRU eviction)


class _AnchorBank:
    """Dynamic keyframe anchor bank with LRU eviction (§2.5)."""

    def __init__(self, threshold: float, persistence: int, max_size: int) -> None:
        self.threshold = threshold
        self.persistence = persistence
        self.max_size = max_size
        self.anchors: list[_AnchorEntry] = []
        self._below_count = 0

    @staticmethod
    def _feature(frame_latent: torch.Tensor) -> torch.Tensor:
        """Compute a [C] feature from a [1, C, 1, H, W] single-frame latent."""
        v = frame_latent.float().mean(dim=(0, 2, 3, 4))  # [C]
        return F.normalize(v.unsqueeze(0), dim=1).squeeze(0)

    def _max_cosine_sim(self, feat: torch.Tensor) -> float:
        if not self.anchors:
            return 0.0
        return max(
            F.cosine_similarity(feat.unsqueeze(0), a.feature.unsqueeze(0)).item()
            for a in self.anchors
        )

    def initialize(self, frame_latent: torch.Tensor, frame_idx: int) -> None:
        """Seed the bank with the first anchor (called once for chunk 0)."""
        feat = self._feature(frame_latent)
        self.anchors = [_AnchorEntry(feature=feat, latent=frame_latent.clone(), frame_idx=frame_idx)]

    def update(
        self,
        frame_latent: torch.Tensor,
        frame_idx: int,
        chunk_idx: int,
        force: bool = False,
    ) -> bool:
        """Evaluate scene-change condition; commit a new anchor if triggered.

        force=True bypasses the threshold/persistence check and always adds an
        anchor.  Used for periodic forced insertions so the bank always contains
        recent frames (prevents content drift on long, visually-stable videos).
        Returns True if a new anchor was committed this call.
        """
        feat = self._feature(frame_latent)
        sim = self._max_cosine_sim(feat)
        if sim < self.threshold:
            self._below_count += 1
        else:
            self._below_count = 0

        committed = False
        if force or self._below_count >= self.persistence:
            if force:
                self._below_count = 0  # forced insert resets streak
            else:
                self._below_count = 0
            self.anchors.append(
                _AnchorEntry(
                    feature=feat,
                    latent=frame_latent.clone(),
                    frame_idx=frame_idx,
                    last_retrieved=chunk_idx,
                )
            )
            committed = True
            # LRU eviction: remove the anchor retrieved least recently
            while len(self.anchors) > self.max_size:
                self.anchors.sort(key=lambda a: a.last_retrieved)
                self.anchors.pop(0)
        return committed

    # Anchors whose similarity to the current overlap exceeds this are redundant:
    # the VideoConditionByLatentIndex overlap conditioning already captures them.
    # Injecting redundant anchors at frame_idx=0 alongside the overlap can confuse
    # the model (two contradictory keyframe conditions at the same position).
    _REDUNDANCY_SIM_THRESHOLD: float = 0.85

    def retrieve(
        self, context_latent: torch.Tensor, top_m: int, chunk_idx: int
    ) -> list[_AnchorEntry]:
        """Return top-m non-redundant anchors by cosine similarity to *context_latent*.

        context_latent: [1, C, F, H, W] — uses the last frame as the query.
        Anchors with similarity > _REDUNDANCY_SIM_THRESHOLD are skipped: they are
        nearly identical to the current overlap and the VideoConditionByLatentIndex
        conditioning already handles them.  This prevents injecting a "chunk N-1 last
        frame" anchor that duplicates the frozen overlap and conflicts with it at
        frame_idx=0 of the new chunk.
        Updates last_retrieved for returned anchors (prevents premature eviction).
        """
        if not self.anchors or top_m == 0:
            return []
        query_feat = self._feature(context_latent[:, :, -1:])
        sims = [
            F.cosine_similarity(query_feat.unsqueeze(0), a.feature.unsqueeze(0)).item()
            for a in self.anchors
        ]
        # Keep only anchors that add information the overlap doesn't already provide
        candidates = [
            i for i, s in enumerate(sims) if s < self._REDUNDANCY_SIM_THRESHOLD
        ]
        if not candidates:
            return []
        top_indices = sorted(candidates, key=lambda i: sims[i], reverse=True)[:top_m]
        result = []
        for idx in top_indices:
            self.anchors[idx].last_retrieved = chunk_idx
            result.append(self.anchors[idx])
        return result


# ---------------------------------------------------------------------------
# CLSSState — orchestrates all per-chunk state
# ---------------------------------------------------------------------------


class CLSSState:
    """Persistent CLSS state maintained across all chunk steps.

    Manages the four CLSS components:
    - Streaming Latent Buffer (SLB): the overlap latent passed to the next chunk
    - Per-channel EMA reference for AdaIN correction (§2.3)
    - Band-energy EMA reference for frequency shrinkage (§2.4)
    - Dynamic anchor bank for long-range identity (§2.5)

    Usage::

        clss = CLSSState(config)
        for chunk_latent_frames in chunk_schedule:
            # 1. Build conditionings: overlap (§2.1) + anchors (§2.5)
            conds = clss.get_overlap_conditioning() + clss.get_anchor_conditioning()

            # 2. Run DiffusionStage with *conds*; get chunk output [1,C,F,H,W]
            generated = ...

            # 3. Extract new frames (drop the overlap region)
            new_frames = generated[:, :, overlap_lf:] if chunk_idx > 0 else generated

            # 4. Apply CLSS corrections (§2.3 + §2.4), update EMA + anchor bank
            corrected = clss.post_process(new_frames)

            # 5. Update the SLB for the next iteration
            clss.update_buffer(corrected)

            # 6. Accumulate corrected for final decoding
            all_latents.append(corrected)
    """

    def __init__(self, config: CLSSConfig) -> None:
        self.config = config
        self._ema = _PerChannelEMA()
        self._band_ema = _BandEMARef()
        self._anchor_bank = _AnchorBank(
            threshold=config.anchor_threshold,
            persistence=config.anchor_persistence,
            max_size=config.anchor_max_size,
        )
        # Streaming Latent Buffer: last overlap_latent_frames from the previous chunk
        self._overlap_latent: Optional[torch.Tensor] = None   # [1, C, overlap_F, H, W]
        self._chunk_index: int = 0
        self._abs_frame_idx: int = 0  # next absolute latent frame index to be written
        # §3.7 Open-loop model gain: set by the pipeline after a perturbation experiment
        self.g_measured: Optional[float] = None
        self.rho_closed: Optional[float] = None

    # ------------------------------------------------------------------
    # Step 2  — Overlap conditioning with implicit re-noising (§2.1)
    # ------------------------------------------------------------------

    def get_overlap_conditioning(self) -> list[ConditioningItem]:
        """Return the VideoConditionByLatentIndex that re-noises the overlap context.

        Setting strength = 1 − τc makes the existing GaussianNoiser apply:
            L̃_overlap = (1 − τc)·L̂_overlap + τc·ε
        exactly matching the CLSS forward-diffusion formula (§2.1, Eq. 3).
        The transformer sees those tokens tagged with timestep σ·τc, so they
        are actively denoised rather than held frozen.

        Returns an empty list for the first chunk (no SLB yet).
        """
        if self._overlap_latent is None:
            return []
        return [
            VideoConditionByLatentIndex(
                latent=self._overlap_latent,
                strength=1.0 - self.config.tau_c,
                latent_idx=0,
            )
        ]

    # ------------------------------------------------------------------
    # Step 3  — Anchor conditioning (§2.5)
    # ------------------------------------------------------------------

    def get_anchor_conditioning(self) -> list[ConditioningItem]:
        """Return VideoConditionByKeyframeIndex items for top-m anchors.

        Anchors are placed at frame_idx=0 (the start of the current chunk).
        This is intentional: each chunk uses LOCAL coordinates (0 … chunk_lf-1),
        and VideoConditionByKeyframeIndex shifts token positions by frame_idx.
        Storing the ABSOLUTE frame index and passing it here would place the
        anchor at the wrong temporal position within the chunk (e.g. absolute=41
        would land outside a 23-frame chunk entirely).  frame_idx=0 matches the
        standard image-conditioning usage and lets the model treat the anchor as
        a global style/identity reference for the whole chunk.

        Returns an empty list when the SLB is empty (first chunk) or the
        anchor bank has not yet been populated.
        """
        if self._overlap_latent is None:
            return []
        anchors = self._anchor_bank.retrieve(
            self._overlap_latent, self.config.anchor_top_m, self._chunk_index
        )
        return [
            VideoConditionByKeyframeIndex(
                keyframes=a.latent,
                frame_idx=0,
                strength=self.config.anchor_strength,
            )
            for a in anchors
        ]

    # ------------------------------------------------------------------
    # Steps 5–6 — Post-process generated new frames
    # ------------------------------------------------------------------

    def post_process(self, new_frames: torch.Tensor) -> torch.Tensor:
        """Apply AdaIN (§2.3) and frequency shrinkage (§2.4) to *new_frames*.

        Both corrections are applied to the *new* frames only (the overlap region
        is not post-processed — it was already corrected when it was generated).
        EMA references are updated with the fully corrected output (Algorithm 1,
        step 7).

        new_frames: [1, C, F, H, W]
        Returns corrected latent with the same shape and dtype.
        """
        cfg = self.config
        cidx = self._chunk_index

        if logger.isEnabledFor(logging.DEBUG):
            mu_raw, sig_raw = _chan_stats(new_frames)
            logger.debug(
                "[CLSS] chunk=%d  raw_latent  μ̄=%.4f  σ̄=%.4f  "
                "μ_range=[%.4f, %.4f]  σ_range=[%.4f, %.4f]",
                cidx,
                sum(mu_raw) / len(mu_raw), sum(sig_raw) / len(sig_raw),
                min(mu_raw), max(mu_raw), min(sig_raw), max(sig_raw),
            )

        # §2.3 AdaIN blended toward EMA reference (no-op when EMA uninitialised)
        # Fixed β from config — the paper uses a constant β in [0.20, 0.40].
        # Stability is verified via ρ = (1−β)·α(τ_c)·g < 1 (§3.7), not enforced
        # adaptively; adaptive β over-normalises latents and destroys natural variation.
        _ema_mu_log  = self._ema.mean.mean().item() if self._ema.mean is not None else float("nan")
        _ema_sig_log = self._ema.std.mean().item()  if self._ema.std  is not None else float("nan")
        _pre_mean = new_frames.float().mean().item()
        _pre_std  = new_frames.float().std().item()
        out = self._ema.apply_adain(new_frames, cfg.beta, cfg.adain_max_amplification)
        # §2.3 always-on telemetry: EMA reference + AdaIN correction direction
        print(
            f"[CLSS] chunk={cidx}"
            f"  ema_mu={_ema_mu_log:.4f}  ema_sigma={_ema_sig_log:.4f}"
            f"  adain_delta_mean={out.float().mean().item() - _pre_mean:+.5f}"
            f"  delta_std={out.float().std().item() - _pre_std:+.5f}"
        )

        if logger.isEnabledFor(logging.DEBUG) and self._ema.mean is not None:
            mu_adain, sig_adain = _chan_stats(out)
            ema_mu_mean = sum(self._ema.mean.tolist()) / len(self._ema.mean)
            ema_sig_mean = sum(self._ema.std.tolist()) / len(self._ema.std)
            logger.debug(
                "[CLSS] chunk=%d  after_adain  μ̄=%.4f  σ̄=%.4f  β=%.4f  "
                "ema_ref: μ̄_ema=%.4f  σ̄_ema=%.4f",
                cidx,
                sum(mu_adain) / len(mu_adain), sum(sig_adain) / len(sig_adain),
                cfg.beta, ema_mu_mean, ema_sig_mean,
            )

        # §2.4 Frequency-band soft shrinkage (updates band EMA internally)
        out = _apply_band_soft_shrinkage(
            out, self._band_ema, cfg.n_freq_bands, cfg.freq_gamma, cfg.ema_lambda,
            chunk_idx=cidx,
        )

        # §2.3 Update per-channel EMA with the corrected output (step 7)
        self._ema.update(out, cfg.ema_lambda, sigma_max_drift=cfg.ema_sigma_max_drift)

        if logger.isEnabledFor(logging.DEBUG):
            mu_final, sig_final = _chan_stats(out)
            logger.debug(
                "[CLSS] chunk=%d  post_process_done  μ̄=%.4f  σ̄=%.4f",
                cidx,
                sum(mu_final) / len(mu_final), sum(sig_final) / len(sig_final),
            )

        return out

    # ------------------------------------------------------------------
    # Step 8  — Update SLB and anchor bank
    # ------------------------------------------------------------------

    def update_buffer(self, output_latent: torch.Tensor) -> None:
        """Push the trailing overlap frames to the SLB; update the anchor bank.

        output_latent: [1, C, F, H, W] — corrected new frames from this chunk.
        The last min(overlap_latent_frames, F) frames become the SLB for the
        next chunk.  The anchor bank is updated with the final frame of this
        chunk (scene-change detection, §2.5).
        """
        cfg = self.config
        F_total = output_latent.shape[2]

        # §2.5 Anchor bank: evaluate scene change with the last frame
        last_frame = output_latent[:, :, -1:]  # [1, C, 1, H, W]
        abs_last_frame_idx = self._abs_frame_idx + F_total - 1
        if self._chunk_index == 0:
            self._anchor_bank.initialize(last_frame, abs_last_frame_idx)
            _new_anchor = True
        else:
            forced = (
                cfg.anchor_force_every > 0
                and self._chunk_index % cfg.anchor_force_every == 0
            )
            _new_anchor = self._anchor_bank.update(
                last_frame, abs_last_frame_idx, self._chunk_index, force=forced,
            )

        # §2.5 always-on telemetry: anchor bank events
        _feat_last  = _AnchorBank._feature(last_frame)
        _max_sim    = self._anchor_bank._max_cosine_sim(_feat_last)
        _bank_fids  = [a.frame_idx for a in self._anchor_bank.anchors]
        print(
            f"[CLSS] chunk={self._chunk_index}"
            f"  anchors: bank_size={len(self._anchor_bank.anchors)}"
            f"  last_frame_max_sim={_max_sim:.4f}"
            f"  new_anchor={_new_anchor}"
            f"  scene_change_streak={self._anchor_bank._below_count}"
            f"  bank_frame_ids={_bank_fids}"
        )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[CLSS] chunk=%d  anchor_bank  n_anchors=%d  "
                "last_frame_max_sim=%.4f  below_streak=%d  threshold=%.2f",
                self._chunk_index,
                len(self._anchor_bank.anchors),
                _max_sim,
                self._anchor_bank._below_count,
                cfg.anchor_threshold,
            )

        # SLB: last overlap_latent_frames frames (or all if chunk is smaller)
        n_overlap = min(cfg.overlap_latent_frames, F_total)
        self._overlap_latent = output_latent[:, :, -n_overlap:].clone()

        self._abs_frame_idx += F_total
        self._chunk_index += 1

    # ------------------------------------------------------------------
    # Stability diagnostics (§2.6)
    # ------------------------------------------------------------------

    @property
    def chunk_index(self) -> int:
        """Number of chunks processed so far."""
        return self._chunk_index

    def loop_gain_estimate(self, beta: Optional[float] = None, tau_c: Optional[float] = None) -> float:
        """Estimate the closed-loop gain ρ_loop = (1 − β)·α(τc) (§2.6, Eq. 6).

        For bounded drift we need ρ_loop < 1.  The formula omits the open-loop
        model gain g (which must be measured empirically, §2.6), so the returned
        value is a lower bound on ρ_loop.

        Returns
        -------
        float
            (1 − β) · (1 − τc).  Values well below 1 give large stability margins.
        """
        b = beta if beta is not None else self.config.beta
        t = tau_c if tau_c is not None else self.config.tau_c
        return (1.0 - b) * (1.0 - t)
