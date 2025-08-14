import sounddevice as sd
import numpy as np
import DataAugmentation as Da
import DataPipeline as Dp
from pynput import keyboard
import os

class_labels = sorted(os.listdir(Dp.DATA_PATH))

def record_and_process():
    print("Recording...")
    audio = sd.rec(
        Dp.AUDIO_LENGTH,
        samplerate=Da.SR,
        channels=1,
        dtype='float32'
    )
    sd.wait()
    # Flatten (n_samples, 1) -> (n_samples,)
    audio = np.squeeze(audio)
    processed_audio = Dp.preprocess_live_audio(audio)
    return processed_audio

def predict_from_mfcc(mfcc,model):
    pred = model.predict(mfcc)
    predicted_class = np.argmax(pred)
    predicted_class_label = class_labels[predicted_class]
    confidence = pred[0][predicted_class]
    print(f"Predicted ClassID: {predicted_class} --> Label: {predicted_class_label}  --> Confidence: {confidence}")


def on_press(key):
    try:
        if key == keyboard.Key.space:  # spacebar
            mfcc = record_and_process()
            predict_from_mfcc(mfcc,Dp.load_model())
    except AttributeError:
        pass  # Ignore special keys

