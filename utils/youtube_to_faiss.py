import os
import sys
import json
import argparse
import time

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_path)

import faiss
import torch
import numpy as np
from tqdm import tqdm
import librosa
from torch.utils.data import DataLoader

from utils.utils import extract_mel_spectrogram
from models.neural_fingerprinter import Neural_Fingerprinter
from generation.generate_fingerprints import FileDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='Path to json config file.')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    with open(os.path.join(project_path, args.config), 'r') as f:
        config_file = json.load(f)

    model = Neural_Fingerprinter()
    model.load_state_dict(torch.load(os.path.join(project_path, config_file['weights'])))
    F = config_file['SR']
    H = config_file['hop_size']
    batch_size = config_file['batch_size']

    with open(os.path.join(project_path, config_file['json']), 'r') as f:
        json_correspondence = json.load(f)

    index = faiss.read_index(os.path.join(project_path, config_file['index']))
    faiss_indexes = sorted([int(x) for x in json_correspondence.keys()])
    next_index = index.ntotal

    youtube_links = config_file['youtube']
    names = config_file['song_names']

    wavs = []
    model.eval()
    with torch.no_grad():
        for name, url in tqdm(zip(names, youtube_links), total=len(names)):
            temp_path = os.path.join(project_path, 'temp')

            command = f'yt-dlp -x --audio-format wav --audio-quality 0 --force-overwrites ' +\
                    f'--output {temp_path}.wav ' +\
                    f'--postprocessor-args "-ar {F} -ac 1" ' + url + " --quiet"
            os.system(command)
            time.sleep(4)

            try:
                file_dset = FileDataset(file=temp_path + '.wav', sr=F, hop_size=H)
            except Exception as e:
                print(f'Failed to download {name}')
                raise
            file_dloader = DataLoader(file_dset, batch_size=batch_size, shuffle=False)
            fingerprints = []
            for X in file_dloader:
                X = model(X)
                fingerprints.append(X.numpy())

            fingerprints = np.vstack(fingerprints)
            index.add(fingerprints)
            json_correspondence[next_index] = name
            next_index += fingerprints.shape[0]
            wavs.append(temp_path)

    faiss.write_index(index, os.path.join(project_path, config_file['index']))
    with open(os.path.join(project_path, config_file['json']), 'w') as f:
        json.dump(json_correspondence, f)
