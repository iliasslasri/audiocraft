"""MAGNET Audio Inpainting Pipeline.

Text-guided audio inpainting: masks a temporal region in EnCodec token space
and re-generates it conditioned on a new text description, then evaluates using
CLAP consistency, boundary smoothness, semantic shift, and FAD metrics.
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

from audiocraft.models import MAGNeT
from audiocraft.data.audio import audio_write
from audiocraft.metrics.clap_consistency import CLAPTextConsistencyMetric

from inpainting_metrics import compute_all_sample_metrics, frechet_audio_distance

# Defaults 

MODEL_NAME = "facebook/magnet-medium-30secs"
SAMPLES_DIR = "samples_magnet"
OUTPUT_DIR = "inpainted_magnet_entire_audio"
DESCRIPTIONS_PATH = "data/magnet_descriptions_slight_change.csv"
NUM_SAMPLES = 16
TARGET_DURATION = 30  # seconds

CFG_MAX_SCALES = [1.0, 3.0, 5.0, 10.0, 15.0]
CFG_MIN_SCALES = [0.0, 1.0, 3.0, 5.0, 10.0]
SAVE_PROBABILITY = 0.05  # fraction of samples to save as audio files


# Audio I/O helpers 
def load_and_prepare_audio(path: str, model, target_samples: int, device: str) -> torch.Tensor:
    """Load, resample, channel-match, and pad/trim audio to exact target length.

    Returns tensor of shape [1, C, target_samples] on the given device.
    """
    wav, sr = torchaudio.load(path)
    if sr != model.sample_rate:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=model.sample_rate)(wav)

    # Match channel count
    n_ch = model.compression_model.channels
    if wav.shape[0] > n_ch:
        wav = wav[:n_ch]
    elif wav.shape[0] < n_ch:
        wav = wav.repeat(n_ch, 1)

    # Trim / pad to target length
    if wav.shape[-1] > target_samples:
        wav = wav[..., :target_samples]
    elif wav.shape[-1] < target_samples:
        wav = F.pad(wav, (0, target_samples - wav.shape[-1]))

    return wav.unsqueeze(0).to(device)


# Description selection 
def select_target_descriptions(clap_metric, descriptions, samples_metadata, top_k=3):
    """For each sample, find the top-k most semantically similar (but different) descriptions.

    Uses CLAP text embeddings and cosine similarity to pick descriptions that are
    close to the original but distinct enough to create a meaningful edit.

    Returns a tensor of indices [top_k, num_samples].
    """
    tokenizer = clap_metric._tokenizer

    desc_embs = []
    for desc in tqdm(descriptions, desc="Embedding target descriptions"):
        with torch.no_grad():
            emb = clap_metric.model.get_text_embedding([desc], tokenizer=tokenizer, use_tensor=True)
            desc_embs.append(emb.cpu())
    desc_embs = torch.cat(desc_embs, dim=0)

    orig_embs = []
    for meta in tqdm(samples_metadata, desc="Embedding original descriptions"):
        with torch.no_grad():
            emb = clap_metric.model.get_text_embedding([meta["prompt"]], tokenizer=tokenizer, use_tensor=True)
            orig_embs.append(emb.cpu())
    orig_embs = torch.cat(orig_embs, dim=0)

    # Cosine similarity matrix [num_descriptions x num_samples]
    sim = F.normalize(desc_embs, dim=1) @ F.normalize(orig_embs, dim=1).T
    return torch.topk(sim, k=top_k, dim=0).indices


def inpaint(
    model_name=MODEL_NAME,
    output_dir=OUTPUT_DIR,
    samples_dir=SAMPLES_DIR,
    num_samples=NUM_SAMPLES,
    descriptions_path=DESCRIPTIONS_PATH,
    start_time=10.0,
    end_time=20.0,
    soft_mask_transition=0,
    position_aware_cfg=False,
    batch_size=4,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(output_dir, exist_ok=True)

    #  Load model 
    print(f"Loading {model_name} on {device}...")
    model = MAGNeT.get_pretrained(model_name)
    target_samples = TARGET_DURATION * model.sample_rate
    frame_rate = model.compression_model.frame_rate

    #  Load descriptions 
    import pandas as pd
    descriptions = pd.read_csv(descriptions_path)["caption"].tolist()

    #  Load sample metadata 
    with open(f"{samples_dir}/metadata.json") as f:
        samples_metadata = json.load(f)

    #  Initialize CLAP metric 
    clap = CLAPTextConsistencyMetric(
        model_path="music_audioset_epoch_15_esc_90.14.pt",
        model_arch="HTSAT-base",
        enable_fusion=False,
    ).to(device)

    #  Select target descriptions via CLAP similarity 
    top_k_indices = select_target_descriptions(clap, descriptions, samples_metadata)

    #  Mask parameters 
    gap_start_sample = int(start_time * model.sample_rate)
    gap_end_sample = int(end_time * model.sample_rate)
    mask_start_idx = int(start_time * frame_rate)
    mask_end_idx = int(end_time * frame_rate)

    #  Run inpainting in batches 
    all_metrics = []
    all_generated_wavs = []  # for FAD
    all_reference_wavs = []  # for FAD

    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        current_batch_size = batch_end - batch_start
        
        print(f"\n{'='*60}")
        print(f"Processing batch {batch_start//batch_size + 1} (samples {batch_start} to {batch_end-1})")

        batch_wavs = []
        batch_target_descs = []
        batch_original_descs = []

        for idx in range(batch_start, batch_end):
            original_desc = samples_metadata[idx]["prompt"]
            target_desc = descriptions[top_k_indices[0, idx].item()]
            if target_desc == original_desc:
                target_desc = descriptions[top_k_indices[1, idx].item()]
                
            batch_original_descs.append(original_desc)
            batch_target_descs.append(target_desc)

            wav = load_and_prepare_audio(
                f"{samples_dir}/{idx}.wav.wav", model, target_samples, device
            )
            batch_wavs.append(wav)
            all_reference_wavs.append(wav.squeeze(0).cpu())

        # Combine into batch tensors
        # wav shape from load_and_prepare is [1, C, T], so cat along dim=0 -> [B, C, T]
        batch_wav_tensor = torch.cat(batch_wavs, dim=0)

        # Encode with masked gap
        wav_masked = batch_wav_tensor.clone()
        wav_masked[..., gap_start_sample:gap_end_sample] = 0
        with torch.no_grad():
            prompt_tokens, _ = model.compression_model.encode(wav_masked)

        # Prepare conditions
        attributes, prompt_tokens = model._prepare_tokens_and_attributes(
            batch_target_descs, batch_wav_tensor
        )
        assert prompt_tokens is not None

        # Sweep CFG scales
        with torch.no_grad():
            for cfg_max in tqdm(CFG_MAX_SCALES, desc=f"Batch {batch_start//batch_size + 1} max_cfg", leave=False):
                for cfg_min in CFG_MIN_SCALES:
                    if cfg_min >= cfg_max:
                        continue

                    # Generate
                    output_tokens = model.lm.generate(
                        prompt=prompt_tokens,
                        conditions=attributes,
                        mask_start_idx=mask_start_idx,
                        mask_end_idx=mask_end_idx,
                        num_samples=1,
                        max_gen_len=int(TARGET_DURATION * frame_rate),
                        temp=3.0,
                        max_cfg_coef=cfg_max,
                        min_cfg_coef=cfg_min,
                        soft_mask_transition=soft_mask_transition,
                        position_aware_cfg=position_aware_cfg,
                    )
                    
                    # Decode batch [B, K, S] -> [B, C, T]
                    out_wavs = model.compression_model.decode(output_tokens, None)

                    for b_idx in range(current_batch_size):
                        global_idx = batch_start + b_idx
                        single_out_wav = out_wavs[b_idx:b_idx+1]
                        single_orig_wav = batch_wav_tensor[b_idx:b_idx+1]
                        
                        target_desc = batch_target_descs[b_idx]
                        original_desc = batch_original_descs[b_idx]

                        # Save a subset of audio files for listening
                        if torch.rand(1).item() < SAVE_PROBABILITY:
                            fname = f"{output_dir}/{global_idx}_cfg_{cfg_min}_{cfg_max}"
                            audio_write(fname, single_out_wav[0].cpu(), model.sample_rate, strategy="loudness")

                        # Compute all per-sample metrics
                        metrics = compute_all_sample_metrics(
                            clap, single_orig_wav, single_out_wav,
                            target_desc, original_desc, model.sample_rate,
                            gap_start_sample, gap_end_sample,
                        )
                        metrics["sample_idx"] = global_idx
                        metrics["max_cfg_scale"] = cfg_max
                        metrics["min_cfg_scale"] = cfg_min
                        metrics["original_description"] = original_desc
                        metrics["target_description"] = target_desc
                        all_metrics.append(metrics)

                        # Collect for FAD (one per CFG config per sample)
                        all_generated_wavs.append(single_out_wav.squeeze(0).cpu())

    #  Compute FAD (aggregate metric over all samples) 
    print(f"\nComputing FAD over {len(all_generated_wavs)} generated vs {len(all_reference_wavs)} reference samples...")
    try:
        fad_score = frechet_audio_distance(
            clap, all_generated_wavs, all_reference_wavs, model.sample_rate
        )
        print(f"  FAD = {fad_score:.4f}")
    except Exception as e:
        print(f"  FAD computation failed: {e}")
        fad_score = None

    #  Save results 
    results = {
        "config": {
            "model": model_name,
            "start_time": start_time,
            "end_time": end_time,
            "soft_mask_transition": soft_mask_transition,
            "position_aware_cfg": position_aware_cfg,
            "num_samples": num_samples,
        },
        "fad": fad_score,
        "per_sample_metrics": all_metrics,
    }
    out_path = f"{output_dir}/inpainting_metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="MAGNET Audio Inpainting")
    parser.add_argument("--start", type=float, default=10.0,
                        help="Start time (seconds) for the mask")
    parser.add_argument("--end", type=float, default=20.0,
                        help="End time (seconds) for the mask")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help="Output directory")
    parser.add_argument("--soft_mask_transition", type=int, default=0,
                        help="Transition zone width in tokens (0=hard mask, 50≈1s at 50Hz)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for generating samples")
    parser.add_argument("--position_aware_cfg", action="store_true",
                        help="Position-aware CFG (stronger at gap center)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    inpaint(
        start_time=args.start,
        end_time=args.end,
        output_dir=args.output_dir,
        soft_mask_transition=args.soft_mask_transition,
        position_aware_cfg=args.position_aware_cfg,
        batch_size=args.batch_size,
    )