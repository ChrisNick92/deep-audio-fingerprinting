import os
import sys
import argparse
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', required=True, help='Csv file containing filenames, youtube links.')
    parser.add_argument('-o', '--output_folder', required=True, help='Output folder to store the wav files.')
    parser.add_argument('-sr', '--sampling_rate', type=int, default=8000, help='The sampling rate to use.')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    csv_file = args.file
    output_folder = args.output_folder
    sr = args.sampling_rate

    with open(csv_file, 'r') as f:
        songs_to_download = csv.reader(f)
        for filename, url in songs_to_download:
            filename = filename.replace(" ", "\\ ")
            command = f'yt-dlp -x --audio-format wav --audio-quality 0 ' +\
                      f'--output {os.path.join(output_folder, filename)}.wav ' +\
                      f'--postprocessor-args "-ar {sr} -ac 1" ' + url + " --quiet"
            os.system(command)