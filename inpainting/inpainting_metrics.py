"""
Evaluation metrics for MAGNET audio inpainting.

Metrics:
  - CLAP Text-Audio Consistency (full audio, gap region, baselines)
  - Boundary Smoothness Score (spectral flux at mask edges)
  - Semantic Shift Ratio (CLAP gap target/original ratio)
  - Fréchet Audio Distance (CLAP-based FAD)
"""

import math
import torch
import typing as tp
import numpy as np

from audiocraft.data.audio_utils import convert_audio


# CLAP Helpers
def _clap_score(clap_metric, wav: torch.Tensor, text: str, sample_rate: int) -> float:
    """Compute a single CLAP cosine similarity score, then reset the metric."""
    clap_metric.update(wav, [text], torch.tensor([wav.shape[-1]]), torch.tensor([sample_rate]))
    score = clap_metric.compute()
    clap_metric.reset()
    return float(score)


def clap_consistency_metrics(
    clap_metric,
    original_wav: torch.Tensor,
    inpainted_wav: torch.Tensor,
    target_desc: str,
    original_desc: str,
    sample_rate: int,
    gap_start_sample: int,
    gap_end_sample: int,
) -> tp.Dict[str, float]:
    """Compute CLAP consistency scores for all audio/text combinations.

    Returns scores for:
      - Full inpainted audio vs target/original descriptions
      - Full original audio vs target/original descriptions (baselines)
      - Gap region of inpainted audio vs target/original descriptions
      - Gap region of original audio vs target/original descriptions (baselines)
    """
    inp_gap = inpainted_wav[..., gap_start_sample:gap_end_sample]
    orig_gap = original_wav[..., gap_start_sample:gap_end_sample]

    return {
        # Full audio
        "clap_inpainted_vs_target": _clap_score(clap_metric, inpainted_wav, target_desc, sample_rate),
        "clap_inpainted_vs_original": _clap_score(clap_metric, inpainted_wav, original_desc, sample_rate),
        "clap_original_vs_target": _clap_score(clap_metric, original_wav, target_desc, sample_rate),
        "clap_original_vs_original": _clap_score(clap_metric, original_wav, original_desc, sample_rate),
        # Gap region only
        "clap_gap_inpainted_vs_target": _clap_score(clap_metric, inp_gap, target_desc, sample_rate),
        "clap_gap_inpainted_vs_original": _clap_score(clap_metric, inp_gap, original_desc, sample_rate),
        "clap_gap_original_vs_target": _clap_score(clap_metric, orig_gap, target_desc, sample_rate),
        "clap_gap_original_vs_original": _clap_score(clap_metric, orig_gap, original_desc, sample_rate),
    }


#  Boundary Smoothness 

def _spectral_flux(wav: torch.Tensor, n_fft: int = 2048, hop_length: int = 512) -> torch.Tensor:
    """Frame-level spectral flux (L2 norm of consecutive STFT magnitude diffs)."""
    spec = torch.stft(wav.squeeze(), n_fft=n_fft, hop_length=hop_length,
                      return_complex=True, window=torch.hann_window(n_fft, device=wav.device))
    mag = spec.abs()
    if mag.dim() == 3:  # [C, F, T] → average channels
        mag = mag.mean(dim=0)
    diff = mag[:, 1:] - mag[:, :-1]
    return torch.norm(diff, dim=0)  # [num_frames - 1]


def boundary_smoothness_score(
    inpainted_wav: torch.Tensor,
    original_wav: torch.Tensor,
    sample_rate: int,
    gap_start_sample: int,
    gap_end_sample: int,
    context_seconds: float = 1.0,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> tp.Dict[str, float]:
    """Measure spectral discontinuity at mask boundaries.

    Compares spectral flux in a window around each boundary against the
    original audio's flux at the same positions.
    Lower flux = smoother; ratio ≈ 1.0 = as smooth as original.
    """
    ctx = int(context_seconds * sample_rate)
    results = {}

    for name, bnd in [("left", gap_start_sample), ("right", gap_end_sample)]:
        lo = max(0, bnd - ctx)
        hi = min(inpainted_wav.shape[-1], bnd + ctx)

        inp_flux = _spectral_flux(inpainted_wav[..., lo:hi], n_fft, hop_length)
        orig_flux = _spectral_flux(original_wav[..., lo:hi], n_fft, hop_length)

        results[f"boundary_{name}_peak_flux"] = float(inp_flux.max())
        results[f"boundary_{name}_mean_flux"] = float(inp_flux.mean())
        results[f"boundary_{name}_orig_peak_flux"] = float(orig_flux.max())
        ratio = float(inp_flux.max() / orig_flux.max()) if orig_flux.max() > 1e-8 else float("inf")
        results[f"boundary_{name}_flux_ratio"] = ratio

    results["boundary_avg_peak_flux"] = 0.5 * (
        results["boundary_left_peak_flux"] + results["boundary_right_peak_flux"])
    results["boundary_avg_flux_ratio"] = 0.5 * (
        results["boundary_left_flux_ratio"] + results["boundary_right_flux_ratio"])
    return results


#  Semantic Shift 

def semantic_shift_ratio(
    clap_metric,
    inpainted_wav: torch.Tensor,
    original_wav: torch.Tensor,
    target_desc: str,
    original_desc: str,
    sample_rate: int,
    gap_start_sample: int,
    gap_end_sample: int,
) -> tp.Dict[str, float]:
    """Measure how much gap content shifted toward the target description.

    shift_to_target > 1.0  → gap is more aligned with target than before.
    drift_from_original < 1.0 → gap lost some original semantics (expected).
    """
    inp_gap = inpainted_wav[..., gap_start_sample:gap_end_sample]
    orig_gap = original_wav[..., gap_start_sample:gap_end_sample]

    inp_tgt = _clap_score(clap_metric, inp_gap, target_desc, sample_rate)
    orig_tgt = _clap_score(clap_metric, orig_gap, target_desc, sample_rate)
    inp_orig = _clap_score(clap_metric, inp_gap, original_desc, sample_rate)
    orig_orig = _clap_score(clap_metric, orig_gap, original_desc, sample_rate)

    return {
        "semantic_shift_to_target": inp_tgt / max(orig_tgt, 1e-8),
        "semantic_drift_from_original": inp_orig / max(orig_orig, 1e-8),
        "gap_clap_inpainted_target": inp_tgt,
        "gap_clap_original_target": orig_tgt,
        "gap_clap_inpainted_original": inp_orig,
        "gap_clap_original_original": orig_orig,
    }


#  Unified per-sample metric

def compute_all_sample_metrics(
    clap_metric,
    original_wav: torch.Tensor,
    inpainted_wav: torch.Tensor,
    target_desc: str,
    original_desc: str,
    sample_rate: int,
    gap_start_sample: int,
    gap_end_sample: int,
) -> tp.Dict[str, float]:
    """Compute all per-sample metrics in a single call."""
    m = {}
    m.update(clap_consistency_metrics(
        clap_metric, original_wav, inpainted_wav,
        target_desc, original_desc, sample_rate,
        gap_start_sample, gap_end_sample))
    m.update(boundary_smoothness_score(
        inpainted_wav, original_wav, sample_rate,
        gap_start_sample, gap_end_sample))
    m.update(semantic_shift_ratio(
        clap_metric, inpainted_wav, original_wav,
        target_desc, original_desc, sample_rate,
        gap_start_sample, gap_end_sample))
    return m


#  FAD (Fréchet Audio Distance)

def frechet_audio_distance(
    clap_metric,
    generated_wavs: tp.List[torch.Tensor],
    reference_wavs: tp.List[torch.Tensor],
    sample_rate: int,
) -> float:
    """CLAP-based Fréchet Audio Distance (no TensorFlow needed).

    Fits Gaussians to CLAP embeddings of generated vs. reference sets
    and computes Fréchet distance. Lower = better. Needs ≥ 30 samples for stability.
    """
    from scipy import linalg

    def _embed_batch(wavs: tp.List[torch.Tensor]) -> np.ndarray:
        embs = []
        for wav in wavs:
            if wav.dim() == 3:
                wav = wav.squeeze(0)
            wav_48k = convert_audio(wav.unsqueeze(0), from_rate=sample_rate,
                                    to_rate=48000, to_channels=1).mean(dim=1)
            with torch.no_grad():
                emb = clap_metric.model.get_audio_embedding_from_data(wav_48k, use_tensor=True)
            embs.append(emb.cpu().numpy())
        return np.concatenate(embs, axis=0)

    gen_embs = _embed_batch(generated_wavs)
    ref_embs = _embed_batch(reference_wavs)

    mu_g, sig_g = gen_embs.mean(0), np.cov(gen_embs, rowvar=False)
    mu_r, sig_r = ref_embs.mean(0), np.cov(ref_embs, rowvar=False)

    diff = mu_g - mu_r
    covmean, _ = linalg.sqrtm(sig_g @ sig_r, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff @ diff + np.trace(sig_g + sig_r - 2 * covmean))
