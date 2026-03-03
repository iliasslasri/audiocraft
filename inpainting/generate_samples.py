from audiocraft.models import MAGNeT
from audiocraft.data.audio import audio_write
import json
import random
import pandas as pd

def main(descriptions_csv_path, output_dir):
    model = MAGNeT.get_pretrained('facebook/magnet-medium-30secs')
    batch_size = 8
    song_descriptions = []
    df = pd.read_csv(descriptions_csv_path)
    for idx, row in df.iterrows():
        song_descriptions.append(row['caption'])
    
    descriptions = random.sample(song_descriptions, min(len(song_descriptions), 20))
    for idx in range(0, len(descriptions), batch_size):
        batch_desc = descriptions[idx:idx+batch_size]
        print(f"Generating batch {idx//batch_size + 1} with descriptions: {batch_desc}")
        wav = model.generate(batch_desc, progress=True)
        for i, one_wav in enumerate(wav):
            audio_write(f'{output_dir}/{idx+i}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)

    data_log = []
    for idx, desc in enumerate(descriptions):
        data_log.append({
            "file_path": f"{output_dir}/{idx}.wav",
            "prompt": desc
        })

    with open(f'{output_dir}/metadata.json', 'w') as f:
        json.dump(data_log, f, indent=2)

if __name__=="__main__":
    descriptions_csv_path = 'data/magnet_descriptions.csv' # 'data/song_describer.csv'
    output_dir = 'samples_magnet'
    main(descriptions_csv_path, output_dir)