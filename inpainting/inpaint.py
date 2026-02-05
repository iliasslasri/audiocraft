import torch
import torchaudio
from audiocraft.models import MAGNeT
from audiocraft.data.audio import audio_write


MODEL_NAME = 'facebook/magnet-medium-30secs' 
INPUT_FILE = 'samples/0.wav'
OUTPUT_NAME = '0_long_80'
TEXT_DESCRIPTION = '80s electronic track with melodic synthesizers, catchy beat and groovy bass'

def inpaint(model_name=MODEL_NAME, output_name=OUTPUT_NAME, input_file=INPUT_FILE, text_desc=TEXT_DESCRIPTION):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on {device}...")
    model = MAGNeT.get_pretrained(model_name)

    print(f"Loading audio: {input_file}...")
    wav, sr = torchaudio.load(input_file)

    target_duration = 30
    target_samples = target_duration * model.sample_rate

    # Resample if needed
    if sr != model.sample_rate:
        wav = torchaudio.transforms.Resample(sr, model.sample_rate)(wav)

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
    print("Creating reference file with silence in the middle...")
    wav_with_silence = wav.clone()
    gap_start_sample = 10 * model.sample_rate
    gap_end_sample = 20 * model.sample_rate
    wav_with_silence[..., gap_start_sample:gap_end_sample] = 0
    audio_write(f'input_with_gap_{output_name}', wav_with_silence[0].cpu(), model.sample_rate, strategy="loudness")

    print("Encoding audio to tokens...")
    with torch.no_grad():
        prompt_tokens, _ = model.compression_model.encode(wav_with_silence)

    # Define Mask Indices
    # MAGNET frame rate is usually 50 Hz
    frame_rate = model.compression_model.frame_rate 
    mask_start_idx = int(10.0 * frame_rate) # Start at 10s
    mask_end_idx = int(20.0 * frame_rate)   # End at 20s

    print(f"Inpainting tokens from index {mask_start_idx} to {mask_end_idx}...")

    # Prepare Text Conditions
    attributes, prompt_tokens = model._prepare_tokens_and_attributes([text_desc], wav)
    assert prompt_tokens is not None
    # Run Generation
    # We access model.lm.generate directly to ensure our new arguments 
    # (mask_start_idx, mask_end_idx) are passed correctly
    print("Generating...")
    scales = [1.0, 3.0, 5.0, 10.0, 15.0]
    with torch.no_grad():
        for cfg in scales:
            output_tokens = model.lm.generate(
                prompt=prompt_tokens,        # Pass full tokens as prompt
                conditions=attributes,
                mask_start_idx=mask_start_idx,
                mask_end_idx=mask_end_idx,  
                num_samples=1,
                max_gen_len=1500,            # 30s * 50Hz = 1500 tokens
                temp=3.0,                     # Temperature for creativity
                max_cfg_coef=cfg,
            )

            print("Decoding tokens to audio...")
            out_wav = model.compression_model.decode(output_tokens, None)

            audio_write(f'{output_name}_cfg_{cfg}.wav', out_wav[0].cpu(), model.sample_rate, strategy="loudness")
            print(f"Saved to {output_name}_cfg_{cfg}.wav...")

    print(f"Done!")

if __name__=="__main__":
    inpaint()