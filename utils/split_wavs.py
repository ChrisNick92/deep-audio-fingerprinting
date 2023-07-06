import os
import argparse
import shutil

from utils import crawl_directory
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input_dir', required=True, help='Directory containing wav files to split into train/val'
    )
    parser.add_argument('-n', '--num_vals', type=int, default=5000, help='Number of wav on val set.')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    input_dir = args.input_dir
    num_vals = args.num_vals

    rng = np.random.default_rng(seed=42)
    all_songs = crawl_directory(input_dir, extension='wav')
    print(f'Total songs: {len(all_songs)}')
    val_songs = np.random.choice(all_songs, size=num_vals, replace=False)
    train_songs = set(all_songs).difference(val_songs)
    print(f'Train: {len(train_songs)} | Val: {len(val_songs)}')
    dirs_to_remove = set()

    try:
        train_path = os.path.join(input_dir, "train")
        os.mkdir(train_path)
    except Exception as e:
        print(e)
        raise
    for train_song in train_songs:
        temp_dir = os.path.basename(os.path.dirname(train_song))
        dirs_to_remove.add(os.path.join(input_dir, temp_dir))
        target_dir = os.path.join(train_path, temp_dir)
        try:
            os.mkdir(target_dir)
        except Exception as e:
            pass
        shutil.move(src=train_song, dst=target_dir)
    
    try:
        val_path = os.path.join(input_dir, "val")
    except Exception as e:
        print(e)
        raise

    for val_song in val_songs:
        temp_dir = os.path.basename(os.path.dirname(train_song))
        target_dir = os.path.join(val_path, temp_dir)
        dirs_to_remove.add(os.path.join(input_dir, temp_dir))
        try:
            os.mkdir(target_dir)
        except Exception as e:
            pass
        shutil.move(src=val_song, dst=target_dir)

    # Remove empty dirs
    for dir in dirs_to_remove:
        os.rmdir(dir)