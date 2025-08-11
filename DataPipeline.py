import os
import numpy as np
import librosa
import tensorflow as tf
from librosa.effects import pitch_shift

import DataAugmentation as Da

#Configs
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data')
AUDIO_LENGTH = int(Da.SR * 1.5)  # 1.5 seconds fixed
hop_length = int(Da.SR * 0.010)
n_fft = int(Da.SR * 0.025)
expected_frames = 148

# Map Dataset
def load_and_preprocess(file_path, shift_sec = 0.0, training=True , spec_augmentation = False):
    # Load audio
    audio, _ = librosa.load(file_path.numpy().decode('utf-8'), sr=Da.SR)

    # Fixed length padding/cropping
    if len(audio) < AUDIO_LENGTH:
        audio = np.pad(audio, (0, AUDIO_LENGTH - len(audio)))
    else:
        audio = audio[:AUDIO_LENGTH]

    # Apply time shift only during training
    if training and (shift_sec != 0.0):
        audio = Da.time_shift(audio,shift_sec)

    if training:
        if np.random.rand() < 0.5: audio = Da.add_noise(audio)
        if np.random.rand() < 0.5: audio = Da.pitch_shift(audio ,Da.SR )

    # Extract MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=Da.SR, n_mfcc=Da.N_MFCC ,n_fft = n_fft , hop_length = hop_length)
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)  # Normalize

    if mfcc.shape[1] < expected_frames:
        pad_width = expected_frames - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    elif mfcc.shape[1] > expected_frames:
        mfcc = mfcc[:, :expected_frames]

    # Apply SpecAugment only during training
    if training and spec_augmentation:
        mfcc = Da.spec_augment(mfcc)




    # Add channel dim
    mfcc = mfcc[..., np.newaxis]

    return mfcc.astype(np.float32)


def tf_wrapper(file_path,shift_sec, label, training=True , spec_augmentation = False):
    mfcc = tf.py_function(load_and_preprocess, inp=[file_path,shift_sec, training,spec_augmentation], Tout=tf.float32)
    mfcc.set_shape([Da.N_MFCC,expected_frames, 1])
    return mfcc, label


#test_files
def preprocess_single_file(filepath):
    audio, _ = librosa.load(filepath, sr=Da.SR)

    # Ensure fixed length (1.5s)
    if len(audio) < AUDIO_LENGTH:
        audio = np.pad(audio, (0, AUDIO_LENGTH - len(audio)))
    elif len(audio) > AUDIO_LENGTH:
        audio = audio[:AUDIO_LENGTH]

    # No random augmentations for testing — just extract MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=Da.SR, n_mfcc=Da.N_MFCC ,n_fft = n_fft , hop_length = hop_length)
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)  # Normalize

    if mfcc.shape[1] < expected_frames:
        pad_width = expected_frames - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    elif mfcc.shape[1] > expected_frames:
        mfcc = mfcc[:, :expected_frames]

    mfcc = mfcc[..., np.newaxis]  # Add channel dim
    mfcc = np.expand_dims(mfcc, axis=0)  # Add batch dim

    return mfcc

def map_fn(file,specaugment,shift_secs, label):
    return tf_wrapper(file, shift_secs, label, training=True,spec_augmentation=specaugment)

