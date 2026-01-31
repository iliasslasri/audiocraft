from audiocraft.models import MAGNeT
from audiocraft.data.audio import audio_write

def main():
    model = MAGNeT.get_pretrained('facebook/magnet-medium-30secs') #'facebook/magnet-small-30secs')
    descriptions = ['Earthy tones, environmentally conscious, ukulele-infused, harmonic, breezy, easygoing, organic instrumentation, gentle grooves', 
                    '80s electronic track with melodic synthesizers, catchy beat and groovy bass',
                    'A grand orchestral arrangement with thunderous percussion, epic brass fanfares, and soaring strings, creating a cinematic atmosphere fit for a heroic battle',
                    'Rock with saturated guitars, a heavy bass line and crazy drum break and fills',
                    'Smooth jazz, with a saxophone solo, piano chords, and snare full drums']
    wav = model.generate(descriptions)  # generates 3 samples.

    for idx, one_wav in enumerate(wav):
        audio_write(f'{idx}_long', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)

if __name__=="__main__":
    main()