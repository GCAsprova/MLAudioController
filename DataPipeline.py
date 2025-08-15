import os
import numpy as np
import librosa
import tensorflow as tf

import CONFIG as cfg
import DataAugmentation as Da



def load_and_preprocess(file_path, shift_sec = 0.0, training=True , spec_augmentation = False):
    # Load audio
    audio, _ = librosa.load(file_path.numpy().decode('utf-8'), sr=cfg.SR)

    # Fixed length padding/cropping
    if len(audio) < cfg.AUDIO_LENGTH:
        audio = np.pad(audio, (0, cfg.AUDIO_LENGTH - len(audio)))
    else:
        audio = audio[:cfg.AUDIO_LENGTH]

    # Apply augmentation only during training
    if training and (shift_sec != 0.0):
        audio = Da.time_shift(audio,shift_sec)

    #Randomly apply noise or shift pitch to mfcc
    if training:
        if np.random.rand() < 0.5: audio = Da.add_noise(audio)
        if np.random.rand() < 0.5: audio = Da.pitch_shift(audio ,cfg.SR )

    # Extract MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=cfg.SR, n_mfcc=cfg.N_MFCC ,n_fft = cfg.n_fft , hop_length = cfg.hop_length)
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)  # Normalize

    if mfcc.shape[1] < cfg.expected_frames:
        pad_width = cfg.expected_frames - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    elif mfcc.shape[1] > cfg.expected_frames:
        mfcc = mfcc[:, :cfg.expected_frames]

    # Apply SpecAugment only during training
    if training and spec_augmentation:
        mfcc = Da.spec_augment(mfcc)

    # Add channel dimension
    mfcc = mfcc[..., np.newaxis]

    return mfcc.astype(np.float32)


def tf_wrapper(file_path,shift_sec, label, training=True , spec_augmentation = False):
    mfcc = tf.py_function(load_and_preprocess, inp=[file_path,shift_sec, training,spec_augmentation], Tout=tf.float32)
    mfcc.set_shape([cfg.N_MFCC,cfg.expected_frames, 1])
    return mfcc, label

def map_fn(file,specaugment,shift_secs, label):
    return tf_wrapper(file, shift_secs, label, training=True,spec_augmentation=specaugment)


def preprocess_live_audio(audio):
    if len(audio) < cfg.AUDIO_LENGTH:
        audio = np.pad(audio, (0, cfg.AUDIO_LENGTH - len(audio)))
    elif len(audio) > cfg.AUDIO_LENGTH:
        audio = audio[:cfg.AUDIO_LENGTH]

    # No random augmentations for testing — just extract MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=cfg.SR, n_mfcc=cfg.N_MFCC ,n_fft = cfg.n_fft , hop_length = cfg.hop_length)
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)  # Normalize

    if mfcc.shape[1] < cfg.expected_frames:
        pad_width = cfg.expected_frames - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    elif mfcc.shape[1] > cfg.expected_frames:
        mfcc = mfcc[:, :cfg.expected_frames]

    mfcc = mfcc[..., np.newaxis]  # Add channel dimension
    mfcc = np.expand_dims(mfcc, axis=0)  # Add batch dimension

    return mfcc

def load_model():
    model_path = os.path.join(cfg.MODELS_PATH, cfg.modelname)
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

