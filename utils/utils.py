import os

import numpy as np


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
