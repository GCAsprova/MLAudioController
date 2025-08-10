import os
import numpy as np
import librosa
import tensorflow as tf
import DataAugmentation as Da

#Configs
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data')
AUDIO_LENGTH = int(Da.SR * 1.5)  # 1.5 seconds fixed

# Map Dataset
def load_and_preprocess(file_path, training=True):
    # Load audio
    audio, _ = librosa.load(file_path.numpy().decode('utf-8'), sr=Da.SR)

    # Fixed length padding/cropping
    if len(audio) < AUDIO_LENGTH:
        audio = np.pad(audio, (0, AUDIO_LENGTH - len(audio)))
    else:
        audio = audio[:AUDIO_LENGTH]

    # Apply time shift only during training
    if training:
        audio = Da.time_shift(audio, 0.25)  # Your time shift function

    # Extract MFCC or Mel Spectrogram
    mfcc = librosa.feature.mfcc(y=audio, sr=Da.SR, n_mfcc=Da.N_MFCC)
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)  # Normalize

    # Apply SpecAugment only during training
    if training:
        mfcc = Da.spec_augment(mfcc)  # Your SpecAugment function

    # Add channel dim
    mfcc = mfcc[..., np.newaxis]

    return mfcc.astype(np.float32)


def tf_wrapper(file_path, label, training=True):
    mfcc = tf.py_function(load_and_preprocess, inp=[file_path, training], Tout=tf.float32)
    mfcc.set_shape([Da.N_MFCC, None, 1])  # Set static shape (adjust None as needed)
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
    mfcc = librosa.feature.mfcc(y=audio, sr=Da.SR, n_mfcc=Da.N_MFCC)
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)  # Normalize
    mfcc = mfcc[..., np.newaxis]  # Add channel dim
    mfcc = np.expand_dims(mfcc, axis=0)  # Add batch dim
    return mfcc


