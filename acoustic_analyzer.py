"""
acoustic_analyzer.py
Acoustic analyser:
 - records audio for 3 seconds
 - applies a window
 - computes FFT
 - shows a spectrogram
"""

import numpy as np
import sounddevice as sd
import scipy.signal as signal
import matplotlib.pyplot as plt


def record_audio(seconds=3, fs=44100):
    """Record audio from the default microphone."""
    print(f"Recording for {seconds} seconds...")
    recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype="float64")
    sd.wait()
    print("Recording complete.")
    return recording.flatten(), fs


def compute_fft(x, fs):
    """Compute FFT of the signal using a Hann window."""
    N = len(x)
    window = np.hanning(N)
    x_windowed = x * window

    X = np.fft.rfft(x_windowed)
    freqs = np.fft.rfftfreq(N, d=1/fs)
    magnitude = np.abs(X) / N
    return freqs, magnitude


def db_scale(magnitude):
    """Convert magnitude to a relative dB scale."""
    magnitude_dB = 20 * np.log10(np.maximum(magnitude, 1e-12))
    # Normalize so the max value is 0 dB
    return magnitude_dB - np.max(magnitude_dB)


def plot_fft(freqs, magnitude_dB):
    plt.figure(figsize=(8, 5))
    plt.semilogx(freqs, magnitude_dB)
    plt.title("Frequency Spectrum (Relative dB)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()


def plot_spectrogram(x, fs):
    f, t, Sxx = signal.spectrogram(x, fs=fs, window="hann", nperseg=1024, noverlap=512)
    plt.figure(figsize=(8, 4))
    plt.pcolormesh(t, f, 10*np.log10(Sxx + 1e-12), shading="gouraud")
    plt.title("Spectrogram (dB)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="dB")
    plt.tight_layout()


def main():
    print("TEST: script reached recording function")
    x, fs = record_audio()

    # Remove DC offset (center the signal)
    x = x - np.mean(x)

    freqs, magnitude = compute_fft(x, fs)
    magnitude_dB = db_scale(magnitude)

    plot_fft(freqs, magnitude_dB)
    plot_spectrogram(x, fs)

    plt.show()


if __name__ == "__main__":
    main()
