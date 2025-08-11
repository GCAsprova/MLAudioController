import os
from datetime import datetime

import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from pathlib import Path



import DataAugmentation as Da
import DataPipeline as Dp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = os.path.join(BASE_DIR, 'models')
model_path = os.path.join(MODELS_PATH, 'model_20250812-004521.h5')
model = tf.keras.models.load_model(model_path)

# Load a new audio file
base_dir = Path(__file__).parent
test_file = base_dir / "test_data"




# Predict
for audio_path in test_file.glob("*.wav"):
    mfcc_input = Dp.preprocess_single_file(str(audio_path))
    pred = model.predict(mfcc_input)
    predicted_class = np.argmax(pred)
    confidence = pred[0][predicted_class]
    print(f"File: {audio_path.name} --> Predicted label: {predicted_class} --> Confidence: {confidence}")
