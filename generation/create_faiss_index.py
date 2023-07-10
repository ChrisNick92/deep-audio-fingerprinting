import os
import sys
import argparse
import json

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_path)

import faiss
from tqdm import tqdm
import numpy as np

from utils.utils import crawl_directory


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--config', required=True, help='The configuration json file.')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    config_file = os.path.join(project_path, args.config)

    with open(config_file, 'r') as f:
        args = json.load(f)

    input_dir = os.path.join(project_path, args['input_dir'])
    output_dir = os.path.join(project_path, args['output_dir'])
    name = args['name']
    index_str = args['index']
    d = args['d']

    if index_str == 'IVF':
        quantizer = faiss.IndexFlatIP(128)
        nlist, M, nbits = 256, 64, 8
        index = faiss.IndexIVFPQ(quantizer, d, nlist, M, nbits)
    else:
        index = faiss.index_factory(d, index_str, faiss.METRIC_INNER_PRODUCT)

    fingerprints = crawl_directory(input_dir, extension='npy')
    total_fingerprints = 0
    for f in tqdm(fingerprints):
        x = np.load(f)
        total_fingerprints += x.shape[0]

    print(f'Total fingerprints: {total_fingerprints}\nCreating index...')
    xb = np.zeros(shape=(total_fingerprints, d), dtype=np.float32)
    json_correspondence = {}

    i = 0
    for f in tqdm(fingerprints):
        x = np.load(f)
        size = x.shape[0]
        xb[i:i + size] = x
        json_correspondence[i] = os.path.basename(f).removesuffix('.npy')
        i += size

    print(f'Training index...')
    index.train(xb)
    print(f'Finished!')
    index.add(xb)

    with open(os.path.join(output_dir, name + '.json'), 'w') as f:
        json.dump(json_correspondence, f)
    faiss.write_index(index, os.path.join(output_dir, name + '.index'))
