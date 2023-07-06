import os
import wave
import random

import numpy as np
from audiomentations import Compose, AddBackgroundNoise, ApplyImpulseResponse
import librosa


def add_background_noise(y: np.ndarray, y_noise: np.ndarray, SNR: float) -> np.ndarray:
    """Apply the background noise y_noise to y with a given SNR
    
    Args:
        y (np.ndarray): The original signal
        y_noise (np.ndarray): The noisy signal
        SNR (float): Signal to Noise ratio (in dB)
        
    Returns:
        np.ndarray: The original signal with the noise added.
    """
    if y.size < y_noise.size:
        y_noise = y_noise[:y.size]
    else:
        y_noise = np.resize(y_noise, y.shape)
    snr = 10**(SNR / 10)
    E_y, E_n = np.sum(y**2), np.sum(y_noise**2)

    z = np.sqrt((E_n / E_y) * snr) * y + y_noise

    return z / z.max()


def crawl_directory(directory: str, extension: str = None) -> list:
    """Crawling data directory
    Args:
        directory (str) : The directory to crawl
    Returns:
        tree (list)     : A list with all the filepaths
    """
    tree = []
    subdirs = [folder[0] for folder in os.walk(directory)]

    for subdir in subdirs:
        files = next(os.walk(subdir))[2]
        for _file in files:
            if extension is not None:
                if _file.endswith(extension):
                    tree.append(os.path.join(subdir, _file))
            else:
                tree.append(os.path.join(subdir, _file))
    return tree


def get_wav_duration(filename: str) -> int:
    """Get the time duration of a wav file"""
    with wave.open(filename, 'rb') as f:
        return f.getnframes() // f.getframerate()


def energy_in_db(signal: np.ndarray) -> float:
    """Return the energy of the input signal in dB.
    
    Args:
        signal (np.ndarray): The input signal.

    Returns:
        float: The energy in dB.
    """
    return 20 * np.log10(np.sum(signal**2))


def time_offset_modulation(signal: np.ndarray, time_index: int, sr: int = 8000, max_offset: float = 0.25) -> np.ndarray:
    """Given an audio segment of signal returns the signal result with a time offset of +- max_offset ms.
    
    Args:
        signal (np.ndarray): The original signal.
        time_index (int): The starting point (i.e. second) of the audio segment.
        max_offset (float): The maximum offset time difference from the original audio segment.

    Return:
        np.ndarray: The signal corresponding to offset of the original audio segment.
    """

    offset = random.choice([random.uniform(-max_offset, -0.1),
                            random.uniform(0.1, max_offset)]) if time_index else random.uniform(0.1, max_offset)
    offset_samples = int(offset * sr)
    start = time_index * sr + offset_samples

    return signal[start:start + sr]


def extract_mel_spectrogram(
    signal: np.ndarray, sr: int = 8000, n_fft: int = 1024, hop_length: int = 256, n_mels: int = 256
) -> np.ndarray:

    S = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    # convert to dB for log-power mel-spectrograms
    return librosa.power_to_db(S, ref=np.max)


def audio_augmentation_chain(
    signal: np.ndarray, time_index: int, noise_path: str, ir_path: str, rng: np.random.Generator, sr: int = 8000
):
    """Given the original clean audio applies a series of audio-augmentation to signal[time_index*sr: (time_index+1)*sr]
    
    Args:
        signal (np.ndarray): The original clean audio.
        time_index (int): The start of the segment corresponding to 1 sec of the original signal.
        noise_path (str): The path containing the wav files corresponding to the noise examples.
        ir_path (str): The path containing the impulse responses to apply.
        rng (np.random.Generator): A np.random.Generator object (required to apply time-offset augmentation)
        sr (int): The sampling rate of the signal.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple corresponding to (Spectrogram_original, Spectrogram_augmented).
    """
    if rng.random() > 0.40:
        augmentation_chain = Compose(
            [
                AddBackgroundNoise(sounds_path=noise_path, min_snr_in_db=5., max_snr_in_db=10., p=1.),
                ApplyImpulseResponse(ir_path=ir_path, p=1.),
            ]
        )
    else:
        augmentation_chain = Compose(
            [
                AddBackgroundNoise(sounds_path=noise_path, min_snr_in_db=0., max_snr_in_db=5., p=1.),
                ApplyImpulseResponse(ir_path=ir_path, p=1.),
            ]
        )
    # Get the corresponding segment
    y = signal[time_index * sr:(time_index + 1) * sr]

    # Offset probability
    if rng.random() > 0.75:
        offset_signal = time_offset_modulation(signal=signal, time_index=time_index)
        augmented_signal = augmentation_chain(offset_signal, sample_rate=8000)
    else:
        augmented_signal = augmentation_chain(y, sample_rate=8000)

    # Clean signal & augmented segments
    S1 = extract_mel_spectrogram(y)
    S2 = extract_mel_spectrogram(augmented_signal)

    return S1, S2

def cutout_spec_augment_mask(rng: np.random.Generator = None):

    H, W = 256, 32
    H_max, W_max = H // 2, int(0.9 * W)
    mask = np.ones((1, H, W), dtype=np.float32)

    rng = rng if rng else np.random.default_rng()
    H_start, dH = rng.integers(low=0, high=H_max, size=2)
    W_start = rng.integers(low=0, high=W_max, size=1).item()
    dW = rng.integers(low=0, high=int(0.1 * W), size=1).item()
    
    mask[:, H_start:H_start + dH, W_start:W_start + dW] = 0

    return mask