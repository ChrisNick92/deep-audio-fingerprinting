import os
import argparse
import subprocess

from utils import crawl_directory
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--source_dir', required=True, help='Input folder.')
    parser.add_argument('-dst', '--destination_dir', required=True, help='Output folder.')
    parser.add_argument('-ex', '--input_extension', default='mp3', help='The file extension of the input audio files.')
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    src, dst = args.source_dir, args.destination_dir
    in_extension = args.input_extension if args.input_extension else 'wav'
    metadata = args.metadata

    input_files = crawl_directory(directory=src, extension=in_extension)
    totals, fails = 0, 0
    failed_files = []

    for file in tqdm(input_files, desc='Converting files.'):
        try:
            out_file = os.path.join(dst, os.path.basename(file).removesuffix(in_extension) + 'wav')
            ffmpeg_command = f"ffmpeg -i {file} -ar 8000 -ac 1 -loglevel error {out_file}"
            subprocess.check_call(ffmpeg_command.split())
        except subprocess.CalledProcessError as e:
            print(f'An error occured on {os.path.basename(file)}\nError: {e}')
            fails += 1
            failed_files.append(file)
            continue
        totals += 1
    print(f'Total songs: {totals}\nFails: {fails}')

    if failed_files:
        print("\n\nFailed files:\n")
        for failed_file in failed_files:
            print(failed_file)