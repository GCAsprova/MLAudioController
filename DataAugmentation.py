import numpy as np
import librosa

#Configs
SR = 48000
N_MFCC = 40
MAX_MASK_PCT = 0.1
N_FREQ_MASKS = 1
N_TIME_MASKS = 1

#Data-Augmentation

def time_shift(audio, shift_sec):
    shift_samples = int(shift_sec * SR)
    if shift_samples > 0:
        audio = np.r_[np.zeros(shift_samples), audio[:-shift_samples]]
    elif shift_samples < 0:
        audio = np.r_[audio[-shift_samples:], np.zeros(-shift_samples)]
    return audio


def extract_mfcc(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=N_MFCC)
    return mfcc


def spec_augment(mfcc):
    mfcc_aug = mfcc.copy()
    num_mel_channels = mfcc.shape[0]
    num_time_steps = mfcc.shape[1]

    # Frequency masking
    for _ in range(N_FREQ_MASKS):
        f = int(np.random.uniform(0, MAX_MASK_PCT) * num_mel_channels)
        f0 = np.random.randint(0, num_mel_channels - f)
        mfcc_aug[f0:f0 + f, :] = 0

    # Time masking
    for _ in range(N_TIME_MASKS):
        t = int(np.random.uniform(0, MAX_MASK_PCT) * num_time_steps)
        t0 = np.random.randint(0, num_time_steps - t)
        mfcc_aug[:, t0:t0 + t] = 0

    return mfcc_aug

def pitch_shift(audio, sr, max_steps=2.0):
    steps = np.random.uniform(-max_steps, max_steps)
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=steps)

def add_noise(audio, noise_level=0.005):
    noise = np.random.randn(len(audio)) * noise_level
    return audio + noise