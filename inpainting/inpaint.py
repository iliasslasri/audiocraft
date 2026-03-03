import torch
import torchaudio
from audiocraft.models import MAGNeT
from audiocraft.data.audio import audio_write
import pandas as pd
import json
from audiocraft.metrics.clap_consistency import CLAPTextConsistencyMetric
from tqdm import tqdm
import torch.nn.functional as F

MODEL_NAME = 'facebook/magnet-medium-30secs' 
SAMPLES_DIR = 'inpainted_magnet_entire_audio'
OUTPUT_DIR = 'inpainted_samples_magnet_slight_change_min_max_cfg'  # 'inpainted_samples_magnet'
DESCRIPTIONS_PATH = 'data/magnet_descriptions_slight_change.csv' # 'data/magnet_augmented_2.csv'
NUM_SAMPLES = 16

def compute_metrics(metric, original_wav, inpainted_wav, target_desc, original_desc, sample_rate, gap_start_sample, gap_end_sample, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Compute CLAP consistency scores for the inpainted audio against both the target and original descriptions.
    """
    clap_consistency = metric
    clap_consistency.update(inpainted_wav, [target_desc], torch.tensor([inpainted_wav.shape[-1]]), torch.tensor([sample_rate]))
    consistency_score_targetdesc = clap_consistency.compute()
    clap_consistency.reset()

    # with original description
    clap_consistency.update(inpainted_wav, [original_desc], torch.tensor([inpainted_wav.shape[-1]]), torch.tensor([sample_rate]))
    consistency_score_originaldesc = clap_consistency.compute()
    clap_consistency.reset()

    # with original wav and target description
    clap_consistency.update(original_wav, [target_desc], torch.tensor([original_wav.shape[-1]]), torch.tensor([sample_rate]))
    consistency_score_original_targetdesc = clap_consistency.compute()
    clap_consistency.reset()

    clap_consistency.update(original_wav, [original_desc], torch.tensor([original_wav.shape[-1]]), torch.tensor([sample_rate]))
    consistency_score_original_originaldesc = clap_consistency.compute()
    clap_consistency.reset()

    # now let's compute just with the gap region to see if the inpainted part is consistent with the target description
    inpainted_gap_region = inpainted_wav[..., gap_start_sample:gap_end_sample]
    clap_consistency.update(inpainted_gap_region, [target_desc], torch.tensor([inpainted_gap_region.shape[-1]]), torch.tensor([sample_rate]))
    consistency_score_gap_targetdesc = clap_consistency.compute()
    clap_consistency.reset()

    # and with original description
    clap_consistency.update(inpainted_gap_region, [original_desc], torch.tensor([inpainted_gap_region.shape[-1]]), torch.tensor([sample_rate]))
    consistency_score_gap_originaldesc = clap_consistency.compute()
    clap_consistency.reset()

    original_gap_region = original_wav[..., gap_start_sample:gap_end_sample]
    clap_consistency.update(original_gap_region, [target_desc], torch.tensor([original_gap_region.shape[-1]]), torch.tensor([sample_rate]))
    consistency_score_gap_original_targetdesc = clap_consistency.compute()
    clap_consistency.reset()

    # and with original description
    clap_consistency.update(original_gap_region, [original_desc], torch.tensor([original_gap_region.shape[-1]]), torch.tensor([sample_rate]))
    consistency_score_gap_original_originaldesc = clap_consistency.compute()
    clap_consistency.reset()

    metrics = {
        "original_description": original_desc,
        "target_description": target_desc,
        "clap_consistency_targetdesc": consistency_score_targetdesc,
        "clap_consistency_originaldesc": consistency_score_originaldesc,
        # original_wav with target/original description
        "clap_consistency_original_targetdesc": consistency_score_original_targetdesc,
        "clap_consistency_original_originaldesc": consistency_score_original_originaldesc,

        # only gap
        "clap_consistency_gap_targetdesc": consistency_score_gap_targetdesc,
        "clap_consistency_gap_originaldesc": consistency_score_gap_originaldesc,

        # gap from original wav as reference
        "clap_consistency_gap_original_targetdesc": consistency_score_gap_original_targetdesc,
        "clap_consistency_gap_original_originaldesc": consistency_score_gap_original_originaldesc,
    }
    return metrics

def inpaint(model_name=MODEL_NAME, output_dir=OUTPUT_DIR, samples_dir=SAMPLES_DIR, num_samples=NUM_SAMPLES, descriptions_path=DESCRIPTIONS_PATH):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on {device}...")
    model = MAGNeT.get_pretrained(model_name)

    descriptions = []
    df = pd.read_csv(descriptions_path)
    for _, row in df.iterrows():
        descriptions.append(row['caption'])
    
    target_duration = 30
    target_samples = target_duration * model.sample_rate

    # load all the audio file in samples_dir
    print(f"Loading audio files from {samples_dir}...")

    json_metadata_path = f'{samples_dir}/metadata.json'
    with open(json_metadata_path, 'r') as f:
        samples_metadata = json.load(f)

    all_metrics = []

    metric = CLAPTextConsistencyMetric(model_path="music_audioset_epoch_15_esc_90.14.pt", model_arch="HTSAT-base", enable_fusion=False).to(device)

    # instead of having random descriptions, we will use the one that is semantically 
    # close to the original description but different enough to require a change in the 
    # audio. This way we can better evaluate if the inpainting is consistent with the 
    # target description and not just any random description. We can use CLAP similarity for this
    # import ipdb

    # ipdb.set_trace()
    # clap_text_encoder = metric.model.model.text_encoder
    tokenizer = metric._tokenizer
    description_embeddings = []
    for desc in tqdm(descriptions, desc="Computing description embeddings"):
        with torch.no_grad():
            text_emb = metric.model.get_text_embedding([desc], tokenizer=tokenizer, use_tensor=True)
            description_embeddings.append(text_emb.cpu())
    description_embeddings = torch.cat(description_embeddings, dim=0)  # (num_descriptions, embedding_dim)

    original_description_embeddings = []
    for metadata in tqdm(samples_metadata, desc="Computing original description embeddings"):
        original_desc = metadata['prompt']
        with torch.no_grad():
            text_emb = metric.model.get_text_embedding([original_desc], tokenizer=tokenizer, use_tensor=True)
            original_description_embeddings.append(text_emb.cpu())
    original_description_embeddings = torch.cat(original_description_embeddings, dim=0)  # (num_samples, embedding_dim)

    desc_norm = F.normalize(description_embeddings, p=2, dim=1)
    orig_norm = F.normalize(original_description_embeddings, p=2, dim=1)
    similarity_matrix = desc_norm @ orig_norm.T  # (num_descriptions, num_samples)
    top_k = 3
    top_k_indices = torch.topk(similarity_matrix, k=top_k, dim=0).indices  # (top_k, num_samples)
    
    for idx in range(num_samples):
        original_desc = samples_metadata[idx]['prompt']
        desc = descriptions[top_k_indices[0, idx].item()]
        if desc == original_desc:
            desc = descriptions[top_k_indices[1, idx].item()]
        
        print(f"\nProcessing sample {idx} with \noriginal description: '{original_desc}' and \ntarget description: '{desc}'")
        wav_path = f'{samples_dir}/{idx}.wav.wav'
        wav, sr = torchaudio.load(wav_path)
        if sr != model.sample_rate:
            wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=model.sample_rate)(wav)
    
        if wav.shape[0] > model.compression_model.channels:
            wav = wav[:model.compression_model.channels]
        elif wav.shape[0] < model.compression_model.channels:
            wav = wav.repeat(model.compression_model.channels, 1)

        # Trim to exactly 30s
        if wav.shape[-1] > target_samples:
            wav = wav[..., :target_samples]
        elif wav.shape[-1] < target_samples:
            # Pad with zeros if too short
            wav = torch.nn.functional.pad(wav, (0, target_samples - wav.shape[-1]))

        wav = wav.unsqueeze(0).to(device)

        # Create the Zeroed version
        wav_with_silence = wav.clone()
        start = 0.0
        end = 30.0
        gap_start_sample = int(start * model.sample_rate)
        gap_end_sample = int(end * model.sample_rate)
        wav_with_silence[..., gap_start_sample:gap_end_sample] = 0

        with torch.no_grad():
            prompt_tokens, _ = model.compression_model.encode(wav_with_silence)

        # Define Mask Indices
        # MAGNET frame rate is usually 50 Hz
        frame_rate = model.compression_model.frame_rate 
        mask_start_idx = int( start* frame_rate)
        mask_end_idx = int(end * frame_rate)

        # Prepare Text Conditions
        attributes, prompt_tokens = model._prepare_tokens_and_attributes([desc], wav)
        assert prompt_tokens is not None
        # Run Generation
        # We access model.lm.generate directly to ensure our new arguments 
        # (mask_start_idx, mask_end_idx) are passed correctly
        scales = [1.0, 3.0, 5.0, 10.0, 15.0]
        with torch.no_grad():
            for cfg in tqdm(scales, desc="Processing CFG scales"):
                for min_cfg in tqdm([0.0, 1.0, 3.0, 5.0, 10.0], desc="Processing min CFG scales"):
                    if min_cfg >= cfg:
                        continue
                    output_tokens = model.lm.generate(
                        prompt=prompt_tokens,
                        conditions=attributes,
                        mask_start_idx=mask_start_idx,
                        mask_end_idx=mask_end_idx,  
                        num_samples=1,
                        max_gen_len=1500,            # 30s * 50Hz = 1500 tokens
                        temp=3.0,                     # Temperature for creativity
                        max_cfg_coef=cfg,
                        min_cfg_coef=min_cfg
                    )

                    out_wav = model.compression_model.decode(output_tokens, None)
                    if torch.randint(0, 100, (1,)).item() < 5:  # Save a few samples for listening
                        audio_write(f'{output_dir}/{idx}_cfg_{min_cfg}_{cfg}', out_wav[0].cpu(), model.sample_rate, strategy="loudness")
                        print(f"Saved to {output_dir}/{idx}_cfg_{min_cfg}_{cfg}.wav...")

                    metrics = compute_metrics(metric, wav.to(device), out_wav.to(device), desc, original_desc, model.sample_rate, gap_start_sample, gap_end_sample, device=device)
                    all_metrics.append({
                        "file_path": f'{output_dir}/{idx}_cfg_{min_cfg}_{cfg}.wav',
                        "original_description": original_desc,
                        "target_description": desc,
                        "max_cfg_scale": cfg,
                        "min_cfg_scale": min_cfg,
                        "consistency_score_targetdesc": metrics["clap_consistency_targetdesc"],
                        "consistency_score_originaldesc": metrics["clap_consistency_originaldesc"],
                        "consistency_score_original_targetdesc": metrics["clap_consistency_original_targetdesc"],
                        "consistency_score_original_originaldesc": metrics["clap_consistency_original_originaldesc"],

                        # with gap region only
                        "consistency_gap_score_targetdesc": metrics["clap_consistency_gap_targetdesc"],
                        "consistency_gap_score_originaldesc": metrics["clap_consistency_gap_originaldesc"],
                        "consistency_gap_original_targetdesc": metrics["clap_consistency_gap_original_targetdesc"],
                        "consistency_gap_original_originaldesc": metrics["clap_consistency_gap_original_originaldesc"],
                    })
    
    # Save metrics to json
    with open(f'{output_dir}/inpainting_metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=4)
    # metrics_df = pd.DataFrame(all_metrics)
    # metrics_df.to_csv(f'{output_dir}/inpainting_metrics.csv', index=False)
    # print(f"Done!")

if __name__=="__main__":
    inpaint()