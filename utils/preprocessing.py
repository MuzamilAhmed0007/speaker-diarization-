# utils/preprocessing.py

import numpy as np
import librosa
import scipy.signal
import soundfile as sf

def adaptive_noise_reduction(audio, sr=16000, noise_floor_db=-30):
    """
    Adaptive noise reduction by spectral gating.
    """
    # Short-Time Fourier Transform
    stft = librosa.stft(audio)
    magnitude, phase = librosa.magphase(stft)

    # Estimate noise threshold dynamically
    noise_estimate = np.percentile(magnitude, 25, axis=1, keepdims=True)
    noise_gate = np.maximum(magnitude - noise_estimate, 0)

    # Normalize and reconstruct
    cleaned_stft = noise_gate * phase
    denoised_audio = librosa.istft(cleaned_stft)
    return denoised_audio

def simulate_beamforming(audio_signals, delays=None, weights=None):
    """
    Basic beamforming: Weighted sum of multi-channel audio signals.
    audio_signals: np.ndarray of shape (channels, samples)
    """
    M, T = audio_signals.shape

    if delays is None:
        delays = [0 for _ in range(M)]
    if weights is None:
        weights = [1 / M for _ in range(M)]

    beamformed = np.zeros(T)

    for m in range(M):
        delayed = np.roll(audio_signals[m], delays[m])
        beamformed += weights[m] * delayed

    # Normalize to avoid clipping
    beamformed = beamformed / np.max(np.abs(beamformed))
    return beamformed

def preprocess_audio(file_path, sr=16000, apply_beamform=True):
    """
    Full preprocessing pipeline.
    1. Load
    2. Denoise
    3. Beamform (simulated if multi-channel)
    """
    audio, sr = librosa.load(file_path, sr=sr)
    print(f"[INFO] Original audio loaded: {file_path}, length = {len(audio)/sr:.2f}s")

    # Denoising
    denoised = adaptive_noise_reduction(audio, sr)
    print(f"[INFO] Denoising completed.")

    # If mono, duplicate channel for simulation
    if len(denoised.shape) == 1:
        denoised_multi = np.vstack([denoised, np.roll(denoised, 5)])
    else:
        denoised_multi = denoised

    # Beamforming (simulated)
    if apply_beamform:
        beamformed = simulate_beamforming(denoised_multi)
        print(f"[INFO] Beamforming completed.")
    else:
        beamformed = denoised

    return beamformed
