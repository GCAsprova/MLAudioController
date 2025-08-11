import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from pathlib import Path
import DataAugmentation as Da
import DataPipeline as Dp

SHIFT_TIMES = [0.0,+0.2,-0.2,0.1,-0.1]
shifted_files = []
shifted_labels = []
shift_secs = []
augmented_files = []
augmented_labels = []
augmented_secs = []
specaugment_flags = []
num_augmentations = 6

#Load File paths and Labels

class_labels = sorted(os.listdir(Dp.DATA_PATH))
label_to_index = {label: idx for idx, label in enumerate(class_labels)}
print(class_labels, label_to_index)
filepaths, labels = [], []
for label in class_labels:
    class_dir = os.path.join(Dp.DATA_PATH, label)
    for file in os.listdir(class_dir):
        if file.endswith(".wav"):
            filepaths.append(os.path.join(class_dir, file))
            labels.append(label_to_index[label])

# Train-test split
train_files, test_files, train_labels, test_labels = train_test_split(
    filepaths, labels, test_size=0.2, stratify=labels, random_state=42
)

for f,l in zip(train_files, train_labels):
    for shift in SHIFT_TIMES:
        shifted_files.append(f)
        shifted_labels.append(l)
        shift_secs.append(shift)

for f, l ,s in zip(shifted_files, shifted_labels, shift_secs):
    # add original (no augmentation)
    augmented_files.append(f)
    augmented_labels.append(l)
    augmented_secs.append(s)
    specaugment_flags.append(False)# False means no SpecAugment
    # add multiple SpecAugment versions
    for _ in range(num_augmentations):
        augmented_files.append(f)
        augmented_labels.append(l)
        augmented_secs.append(s)
        specaugment_flags.append(True)# True means SpecAugment

# Build Dataset
train_ds = tf.data.Dataset.from_tensor_slices((augmented_files,specaugment_flags,augmented_secs, augmented_labels))
train_ds = train_ds.map(Dp.map_fn, num_parallel_calls=tf.data.AUTOTUNE)
#train_ds = train_ds.map(lambda files, labels: Dp.tf_wrapper(files, labels, training=True), num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(750).batch(8).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((test_files, test_labels))
test_ds = test_ds.map(lambda fi, la: Dp.tf_wrapper(fi,0.0, la, training=False), num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(8).prefetch(tf.data.AUTOTUNE)

# First, find width of MFCC from one sample
sample_audio, _ = librosa.load(filepaths[0], sr=Da.SR)
sample_mfcc = Da.extract_mfcc(sample_audio)
MFCC_WIDTH = sample_mfcc.shape[1]

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(Da.N_MFCC, MFCC_WIDTH, 1)),

    # 1.CONV
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),

    # 2.CONV
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),

    # 3.CONV
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.3),

    # Flatten or GAP
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(len(class_labels), activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=30
)

# Load a new audio file
base_dir = Path(__file__).parent
test_file = base_dir / "test_data"




# Predict
for audio_path in test_file.glob("*.wav"):
    mfcc_input = Dp.preprocess_single_file(str(audio_path))
    pred = model.predict(mfcc_input)
    predicted_class = np.argmax(pred)
    confidence = pred[0][predicted_class]
    print(f"File: {audio_path.name} --> Predicted label: {class_labels[predicted_class]} --> Confidence: {confidence}")
