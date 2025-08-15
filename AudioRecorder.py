import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress INFO logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" # optional, disable oneDNN notices

import sounddevice as sd
import numpy as np
import time
from pynput import keyboard

import DataPipeline as Dp
import CONFIG as cfg

class_labels = sorted(os.listdir(cfg.DATA_PATH))
saved_model = Dp.load_model()
keyboard = keyboard.Controller()

def record_and_process():
    print("Recording...")
    audio = sd.rec(
        cfg.AUDIO_LENGTH,
        samplerate=cfg.SR,
        channels=1,
        dtype='float32'
    )
    sd.wait()
    # Flatten (n_samples, 1) -> (n_samples,)
    audio = np.squeeze(audio)
    processed_audio = Dp.preprocess_live_audio(audio)
    return processed_audio

def predict_from_mfcc(mfcc,model):
    pred = model.predict_on_batch(mfcc)
    predicted_class = np.argmax(pred)
    predicted_class_label = class_labels[predicted_class]
    confidence = pred[0][predicted_class]
    print(f"Predicted ClassID: {predicted_class} --> Label: {predicted_class_label}  --> Confidence: {confidence}")
    return predicted_class


def on_press(key):
    try:
        if key == cfg.RECORD_BUTTON:
            mfcc = record_and_process()
            prediction_class = predict_from_mfcc(mfcc,saved_model)
            press_key_for_class(prediction_class)

    except AttributeError:
        pass  # Ignore special keys


def press_key_for_class(class_id):
    """Press mapped key for given class ID"""
    key = cfg.CLASS_TO_KEY.get(class_id)
    if key:
        keyboard.press(key)
        time.sleep(0.05)  # short press
        keyboard.release(key)
        #print(f"[ACTION] Class {class_id} → Key '{key}'")
    else:
        print(f"[WARNING] No key mapped for class {class_id}")