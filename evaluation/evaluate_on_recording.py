import os
import sys
import argparse
import json
import time

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_path)

from models.neural_fingerprinter import Neural_Fingerprinter
from utils.utils import extract_mel_spectrogram, get_winner

import faiss
import torch
import numpy as np
import librosa
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='The configuration file of the recording test.')

    return parser.parse_args()


def find_songs_intervals(csv_file):
    d = {}
    start = 0
    for file in csv_file:
        y, sr = librosa.load(file, sr=8000)
        dur = y.size / sr
        d[os.path.basename(file)] = {'start': start, 'end': start + dur}
        start += dur
    return d


def print_results(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', labels=list(set(y_true)), zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', labels=list(set(y_true)), zero_division=0)
    print(
        f"\nAccuracy score: {acc*100:.2f}%\nPrecision score: {precision*100:.2f}%" +
        f"\nRecall score: {recall*100:.2f}%\n"
    )


if __name__ == '__main__':

    args = parse_args()
    config_file = os.path.join(project_path, args.config)
    with open(config_file, 'r') as f:
        args = json.load(f)
        print(f'Config:\n{args}\n')

    index = faiss.read_index(os.path.join(project_path, args['index']))

    with open(os.path.join(project_path, args['json']), 'r') as f:
        json_correspondence = json.load(f)
        sorted_arr = np.sort(np.array(list(map(int, json_correspondence.keys()))))

    with open(os.path.join(project_path, args['csv']), 'r') as f:
        true_songs = [os.path.join(project_path, args['test_songs'], x.rstrip()) for x in f.readlines()]

    dur = args['duration']
    attention=args['attention']
    device = 'cuda' if torch.cuda.is_available() and args['device'] == 'cuda' else 'cpu'

    model = Neural_Fingerprinter(attention=attention).to(device)
    model.load_state_dict(torch.load(os.path.join(project_path, args['model']), map_location=device))
    print(f'Running on {device}')

    F = args['sr']
    H = args['hop_size']
    k = args['neighbors']
    index.nprobe = args['nprobes']

    songs_intervals = find_songs_intervals(true_songs)
    recording, sr = librosa.load(os.path.join(project_path, args['recording']), sr=F)
    y_true, y_pred = [], []
    inference_time, query_time = [], []

    model.eval()
    with torch.no_grad():
        for song in tqdm(true_songs, desc='Processing songs'):

            label = os.path.basename(song)
            start, end = songs_intervals[label]['start'], songs_intervals[label]['end']
            iters = int((end - start) / dur)
            q, r = divmod(start * sr, sr)
            start = int(q * sr) + int(r)

            for seg in range(iters):

                # Slice recording
                rec_slice = recording[start + seg * dur * F:start + (seg + 1) * dur * F]

                # Inference
                tic = time.perf_counter()
                J = int(np.floor((rec_slice.size - F) / H)) + 1
                xq = [np.expand_dims(extract_mel_spectrogram(rec_slice[j * H:j * H + F]), axis=0) for j in range(J)]
                xq = np.stack(xq)
                out = model(torch.from_numpy(xq).to(device))
                inference_time.append(1000 * (time.perf_counter() - tic))

                # Retrieval
                tic = time.perf_counter()
                D, I = index.search(out.cpu().numpy(), k)
                pred, score = get_winner(json_correspondence, I, D, sorted_arr)
                query_time.append(1000 * (time.perf_counter() - tic))

                y_true.append(label.removesuffix('.wav'))
                y_pred.append(pred)

    print_results(y_true, y_pred)
    total_time = [x + y for x, y in zip(inference_time, query_time)]
    print(f'Inference Time: {np.mean(inference_time)}\nQuery Time: {np.mean(query_time)}\nTotal: {np.mean(total_time)}')
