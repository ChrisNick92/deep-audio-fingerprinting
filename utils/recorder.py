import os
import sys
import argparse
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyaudio


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--duration', type=int, default=10, help='Duration of the recording (sec).')
    parser.add_argument('-o', '--output_path', required=True, help='The path & filename of the output.')

    return parser.parse_args()


# recording hyperparameters
FRAMES_PER_BUFFER = 4000
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 8000

if __name__ == '__main__':

    args = parse_args()
    duration = args.duration
    output = args.output_path

    p = pyaudio.PyAudio()

    stream = p.open(rate=RATE, channels=CHANNELS, format=FORMAT, frames_per_buffer=FRAMES_PER_BUFFER, input=True)
    frames = []

    print(f'Recording starts...')
    for i in range(0, int(RATE / FRAMES_PER_BUFFER * duration)):
        data = stream.read(FRAMES_PER_BUFFER)
        frames.append(data)
    print(f'Steam is closed! Saving file on {output}')

    with wave.open(os.path.join(output) + '.wav', 'wb') as f:
        f.setnchannels(CHANNELS)
        f.setframerate(RATE)
        f.setsampwidth(p.get_sample_size(FORMAT))
        f.writeframes(b"".join(frames))
