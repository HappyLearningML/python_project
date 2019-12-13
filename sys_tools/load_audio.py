#-*-coding:utf-8-*-
from scipy.io import wavfile
import numpy as np

def load_audio(filename, b_normalize=True):
    """Load the audiofile at the provided filename using scipy.io.wavfile.

    Optionally normalizes the audio to the maximum value.

    Parameters
    ----------
    filename : str
        File to load.
    b_normalize : bool, optional
        Normalize to the maximum value.
    """
    _, s = wavfile.read(filename)
    if b_normalize:
        s = s.astype(np.float32)
        s = (s / np.max(np.abs(s)))
        s -= np.mean(s)
    return s