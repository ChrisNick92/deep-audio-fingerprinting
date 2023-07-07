import json
import os
import sys
import argparse

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_path)

from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import numpy as np
import librosa
from tqdm import tqdm
import torch

from utils.utils import crawl_directory, extract_mel_spectrogram
from models.neural_fingerprinter import Neural_Fingerprinter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='The configuration json file.')
    
    return parser.parse_args()

class FileDataset(Dataset):

    def __init__(self, file, sr, hop_size):
        self.y, self.F = librosa.load(file, sr=sr)
        self.H = hop_size

        # Extract spectrograms
        self._get_spectrograms()

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, idx):
        return torch.from_numpy(self.spectrograms[idx])

    def _get_spectrograms(self):
        self.spectrograms = []
        J = int(np.floor((self.y.size - self.F) / self.H)) + 1
        for j in range(J):
            S = extract_mel_spectrogram(signal=self.y[j * self.H:j * self.H + self.F])
            self.spectrograms.append(S.reshape(1, *S.shape))


if __name__ == '__main__':

    # parse args
    args = parse_args()
    config_file = args.config
    with open(config_file, "r") as f:
        args = json.load(f)

    SR = args["SR"]
    HOP_SIZE = args["HOP SIZE"]
    input_dirs = [os.path.join(project_path, dir) for dir in args["input dirs"]]
    output_dir = os.path.join(project_path, args["output dir"])
    batch_size = args["batch size"]
    pt_file = args["weights"]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Neural_Fingerprinter().to(device)
    model.load_state_dict(torch.load(pt_file))
    print(f'Running on {device}')

    all_songs = []
    for dir in input_dirs:
        all_songs += crawl_directory(dir, extension='wav')

    model.eval()
    fails = 0
    totals = len(all_songs)
    p_bar = tqdm(all_songs, desc='Extracting deep audio fingerprints', total=totals)
    with torch.no_grad():
        for file in p_bar:
            file_dset = FileDataset(file=file, sr=SR, hop_size=HOP_SIZE)
            file_dloader = DataLoader(file_dset, batch_size=batch_size, shuffle=False)
            fingerprints = []

            for X in file_dloader:
                X = model(X.to(device))
                fingerprints.append(X.cpu().numpy())

            fingerprints = np.vstack(fingerprints)

            try:
                np.save(
                    file=os.path.join(output_dir,
                                      os.path.basename(file).removesuffix('.wav') + '.npy'),
                    arr=fingerprints
                )
            except Exception as e:
                print(f'Failed to save {os.path.basename(file)} | Error: {e}')
                fails += 1
                continue

    print(f'Totals: {totals}\nFails: {fails}')
