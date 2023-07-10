import argparse
import os
import sys
import json
import time

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_path)

import torch
import faiss
from audiomentations import AddBackgroundNoise
from tqdm import tqdm
import librosa
import numpy as np

from models.neural_fingerprinter import Neural_Fingerprinter
from utils.utils import crawl_directory, extract_mel_spectrogram, query_sequence_search, search_index
from utils.metrics import summary_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--config', required=True, help='The evaluation configuration file.')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    config_file = os.path.join(project_path, args.config)
    with open(config_file, 'r') as f:
        args = json.load(f)

    device = 'cuda' if torch.cuda.is_available() and args['device'] == 'cuda' else 'cpu'
    model = Neural_Fingerprinter().to(device)
    model.load_state_dict(torch.load(os.path.join(project_path, args['model']), map_location=device))
    print(f'Running on {device}')

    index = faiss.read_index(os.path.join(project_path, args['index']))
    with open(os.path.join(project_path, args['json'])) as f:
        json_correspondence = json.load(f)

    sorted_arr = np.sort(np.array(list(map(int, json_correspondence.keys()))))
    snrs = args['snrs']
    durations = args['durations']
    F = args['sr']
    H = args['hop_size']
    k = args['neighbors']
    index.nprobes = args['nprobes']
    
    print(f'\nConfigurations:\nModel: {os.path.basename(args["model"])}\nFaiss vectors: {index.ntotal}\n')

    demo_songs = crawl_directory(os.path.join(project_path, args['demo_songs']), extension='wav')
    background_noises = os.path.join(project_path, args['background_noise'])
    dur_times = {}

    for dur in durations:
        print(f'{10 * "*"} Deep audio test with Query duration: {dur} {10 * "*"}\n')
        top_1_hit_rates = []
        query_times, inference_times = [], []

        for snr in snrs:
            print(f'\n{10 * "*"} SNR: {snr} {10 * "*"}\n')
            b_noise = AddBackgroundNoise(sounds_path=background_noises, min_snr_in_db=snr, max_snr_in_db=snr, p=1.)
            y_true, y_pred = [], []
            offset_true, offset_pred = [], []

            for audio_file in tqdm(demo_songs):

                y, sr = librosa.load(audio_file, sr=F)
                segs = (y.size // F) // dur
                model.eval()

                with torch.no_grad():
                    for seg in range(segs):

                        # Inference
                        tic = time.perf_counter()
                        y_slice = b_noise(y[seg * dur * F:(seg + 1) * dur * F], sample_rate=F)
                        J = int(np.floor((y_slice.size - F) / H)) + 1
                        xq = [
                            np.expand_dims(extract_mel_spectrogram(signal=y_slice[j * H:j * H + F]), axis=0)
                            for j in range(J)
                        ]
                        xq = np.stack(xq)
                        out = model(torch.from_numpy(xq).to(device))
                        inference_times.append(1000 * (time.perf_counter() - tic))

                        # Retrieval
                        tic = time.perf_counter()
                        D, I = index.search(out.cpu().numpy(), k)
                        idx, d = query_sequence_search(D, I)
                        start_idx = search_index(idx, sorted_arr)
                        query_times.append(1000 * (time.perf_counter() - tic))

                        y_true.append(os.path.basename(audio_file).removesuffix('.wav'))
                        y_pred.append(json_correspondence[str(start_idx)])
                        offset_true.append(seg * dur)
                        offset_pred.append((idx - start_idx) * H / F)

            top_1 = summary_metrics(np.array(y_true), np.array(y_pred), np.array(offset_true), np.array(offset_pred))
            top_1_hit_rates.append(top_1)

        total_times = list(map(lambda x: x[0] + x[1], zip(inference_times, query_times)))

        dur_times[dur] = {
            'Inference Time': np.mean(inference_times),
            'Query Time': np.mean(query_times),
            'Total': np.mean(total_times)
        }
        print(dur_times[dur])

    print(f'SNR: {snrs}\nTop 1 hit rate: {top_1_hit_rates}')
    for dur in durations:
        print(f'Dur: {dur} | {dur_times[dur]}')