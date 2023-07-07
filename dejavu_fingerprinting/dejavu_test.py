import os
import argparse
import sys

temp_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(temp_path))

from audiomentations import AddBackgroundNoise
from sklearn.metrics import accuracy_score
import soundfile as sf
import librosa
from tqdm import tqdm
import numpy as np

from dejavu.logic.recognizer.file_recognizer import FileRecognizer
from dejavu import Dejavu
from utils.utils import crawl_directory, get_wav_duration
from utils.metrics import summary_metrics

config = {
    "database": {
        "host": "127.0.0.1",
        "user": "root",
        "password": "mysqlpass",
        "database": "dejavu",
    }
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_folder', required=True, help='Folder with test wav files.')
    parser.add_argument('-bn', '--background_noises', required=True, help='Path containing background noise wav files.')
    parser.add_argument('-d', '--duration', type=int, nargs='+', default=[5], help='Query duration (in sec).')
    parser.add_argument(
        '-snr', '--signal_to_noise_ratio', type=float, nargs='+', default=[5.], help='SNRs (in dB) to add noise.'
    )

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    input_folder = args.input_folder
    noise_folder = args.background_noises
    snrs = args.signal_to_noise_ratio
    durs = args.duration

    audio_files = crawl_directory(input_folder, extension='wav')
    djv = Dejavu(config)
    
    for dur in durs:
        
        print(f'{10 * "*"} Dejavu Test with Query duration: {dur} {10 * "*"}\n')
        top_1_hit_rates = []
        
        for snr in snrs:
            print(f'\n{10 * "*"} SNR: {snr} {10 * "*"}\n')
            
            b_noise = AddBackgroundNoise(sounds_path=noise_folder, min_snr_in_db=snr, max_snr_in_db=snr, p=1)
            y_true, y_pred = [], []
            offset_true, offset_pred = [], []

            for audio_file in tqdm(audio_files):
                
                segs = get_wav_duration(audio_file) // dur
                noise_wav = os.path.join(temp_path, "temp_noise.wav")
                y, sr = librosa.load(audio_file, sr=8000)
                noise = b_noise(y, sample_rate=sr)
                sf.write(file=noise_wav, data=noise, samplerate=sr, subtype='FLOAT')
                temp_wav = os.path.join(temp_path, "temp.wav")
                
                for seg in range(segs):
                    ffmpeg_command = f'ffmpeg -i {noise_wav} -ss {seg * dur} -to {(seg + 1)*dur} ' +\
                        f'{temp_wav} -y -loglevel error'
                    os.system(ffmpeg_command)
                    results = djv.recognize(FileRecognizer, temp_wav)
                    y_pred.append(results['results'][0]['song_name'].decode())
                    y_true.append(os.path.basename(audio_file)[:-4])
                    offset_true.append(seg * dur)
                    offset_pred.append(results["results"][0]['offset_seconds'])

            top_1 = summary_metrics(np.array(y_true), np.array(y_pred), np.array(offset_true), np.array(offset_pred))
            top_1_hit_rates.append(top_1)
            
        print(f'SNR: {snrs}\nTop 1 hit rate: {top_1_hit_rates}')