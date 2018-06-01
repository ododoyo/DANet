import os, sys
libpath = os.path.abspath('..')
sys.path.append(libpath)

import wave
import librosa
import threading
import numpy as np

from numpy.fft import rfft, irfft
from scipy.io import wavfile
from scipy import signal

int16_ampmax = 2 ** 15


def audioread(path, offset=0.0, duration=None, samp_rate=16000):
    signal, sr = librosa.load(path, mono=False, sr=samp_rate,
                              offset=offset, duration=duration)
    return signal.astype(np.float32)


def wavread(path):
    f = wave.open(path)
    channels = f.getnchannels()
    sampwidth = f.getsampwidth()
    assert channels == 1 and sampwidth == 2
    nframes = f.getnframes()
    signal = np.fromstring(f.readframes(nframes), dtype=np.short)
    signal /= int16_ampmax
    return signal.astype(np.float32)


def audiowrite(path, data, samp_rate=16000, normalize=False, threaded=True):
    data = data.copy()
    int16_max = (1 << 15) - 1
    int16_min = -(1 << 15)
    if normalize:
        data /= np.max(np.abs(data))
    data *= int16_ampmax
    sample_clip = np.sum(data < int16_min) + np.sum(data > int16_max)
    data = np.clip(data, int16_min, int16_max)
    data = data.astype(np.int16)
    if threaded:
        threading.Thread(target=wavfile.write,
                         args=(path, samp_rate, data)).start()
    else:
        wavfile.write(path, samp_rate, data)
    return sample_clip


def _samples_to_stft_frames(samples, size, shift):
    if samples <= size - shift:
        return 1
    else:
        return np.ceil((samples - size + shift) / shift).astype(np.int32)


def _stft_frames_to_samples(frames, size, shift):
    return frames * shift + size - shift


def _biorthogonal_window_loopy(analysis_window, shift):
    """
    This version of the synthesis calculation is as close as possible to the
    Matlab impelementation in terms of variable names.
    The results are equal.
    The implementation follows equation A.92 in
    Krueger, A. Modellbasierte Merkmalsverbesserung zur robusten automatischen
    Spracherkennung in Gegenwart von Nachhall und Hintergrundstoerungen
    Paderborn, Universitaet Paderborn, Diss., 2011, 2011
    """
    fft_size = len(analysis_window)
    assert np.mod(fft_size, shift) == 0
    number_of_shifts = len(analysis_window) // shift

    sum_of_squares = np.zeros(shift)
    for synthesis_index in range(0, shift):
        for sample_index in range(0, number_of_shifts + 1):
            analysis_index = synthesis_index + sample_index * shift

            if analysis_index + 1 < fft_size:
                sum_of_squares[synthesis_index] \
                    += analysis_window[analysis_index] ** 2

    sum_of_squares = np.kron(np.ones(number_of_shifts), sum_of_squares)
    synthesis_window = analysis_window / sum_of_squares / fft_size
    return synthesis_window


# compute stft of a 1-dim time_signal
def stft(time_signal, size=1024, shift=256, fading=True,
         window=signal.windows.hann, window_length=None):
    assert time_signal.ndim == 1
    if fading:
        pad = [(size - shift, size - shift)]
        time_signal = np.pad(time_signal, pad, mode='constant')
    frames = _samples_to_stft_frames(time_signal.shape[0], size, shift)
    samples = _stft_frames_to_samples(frames, size, shift)
    pad = [(0, samples - time_signal.shape[0])]
    time_signal = np.pad(time_signal, pad, mode='constant')
    if window_length is None:
        window = window(size)
    else:
        window = window(window_length)
        window = np.pad(window, (0, size - window_length), mode='constant')
    chunk_signal = np.zeros((frames, size))
    for i, j in enumerate(range(0, samples - size + shift, shift)):
        chunk_signal[i] = time_signal[j:j+size]
    return rfft(chunk_signal * window, axis=1)


def istft(stft_signal, size=1024, shift=256, fading=True,
          window=signal.windows.hann, window_length=None):
    assert stft_signal.shape[1] == size // 2 + 1
    if window_length is None:
        window = window(size)
    else:
        window = window(window_length)
        window = np.pad(window, (0, size - window_length), mode='constant')
    window = _biorthogonal_window_loopy(window, shift)
    window *= size
    time_signal = np.zeros(stft_signal.shape[0] * shift + size - shift)
    for i, j in enumerate(range(0, len(time_signal) - size + shift, shift)):
        time_signal[j:j+size] += window * np.real(irfft(stft_signal[i], size))
    if fading:
        time_signal = time_signal[size - shift:len(time_signal) - size + shift]
    return time_signal.astype(np.float32)
