import os
import argparse
import multiprocessing
import shutil
import sys
import pathlib

ABS_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ABS_PATH)

from tqdm import tqdm
import numpy as np

from dejavu import Dejavu
from utils import utils

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
    parser.add_argument('-i', '--input_dir', required=True, help='The input directory with audio files.')
    parser.add_argument('-c', '--chunks', type=int, default=300, help='How many songs to process each time.')
    return parser.parse_args()


if __name__ == '__main__':
    
    args = parse_args()
    input_dir = args.input_dir
    chunks = args.chunks
    num_workers = multiprocessing.cpu_count() // 2

    djv = Dejavu(config=config)

    all_songs = utils.crawl_directory(input_dir, extension='wav')
    print(f'Total songs: {len(all_songs)}')

    num_splits = len(all_songs) // chunks
    songs_splitted = np.array_split(all_songs, indices_or_sections=num_splits)
    temp_folder = os.path.join(input_dir, "temp")

    for songs in tqdm(songs_splitted, desc='Processing split', total=len(songs_splitted)):
        
        os.mkdir(temp_folder)
        for song in songs:
            shutil.copy(song, temp_folder)

        djv.fingerprint_directory(path=temp_folder, extensions=['.wav'], nprocesses=num_workers)
        
        shutil.rmtree(temp_folder)
    
