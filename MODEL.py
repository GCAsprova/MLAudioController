import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.model_selection import StratifiedKFold

import DataPipeline as Dp
import CONFIG as cfg

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress INFO logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" # optional, disable oneDNN notices

skf = StratifiedKFold(n_splits=cfg.k, shuffle=True, random_state=42)

#Load File paths and Labels

class_labels = sorted(os.listdir(cfg.DATA_PATH))
label_to_index = {label: idx for idx, label in enumerate(class_labels)}
print(class_labels, label_to_index)
filepaths, labels = [], []
for label in class_labels:
    class_dir = os.path.join(cfg.DATA_PATH, label)
    for file in os.listdir(class_dir):
        if file.endswith(".wav"):
            filepaths.append(os.path.join(class_dir, file))
            labels.append(label_to_index[label])

filepaths = np.array(filepaths)
labels = np.array(labels)
acc_per_fold = []
loss_per_fold = []

for fold_no, (train_idx , val_idx) in enumerate(skf.split(filepaths,labels),start=1):
    print(f'Training fold {fold_no}')

    train_files = [filepaths[i] for i in train_idx]
    val_files = [filepaths[i] for i in val_idx]

    train_labels = [labels[i] for i in train_idx]
    val_labels = [labels[i] for i in val_idx]

    shifted_files = []
    shifted_labels = []
    shift_secs = []

    augmented_files = []
    augmented_labels = []
    augmented_secs = []
    specaugment_flags = []

    for f,l in zip(train_files, train_labels):
        for shift in cfg.SHIFT_TIMES:
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
        for _ in range(cfg.num_augmentations):
            augmented_files.append(f)
            augmented_labels.append(l)
            augmented_secs.append(s)
            specaugment_flags.append(True)# True means SpecAugment

    # Build Dataset
    train_ds = tf.data.Dataset.from_tensor_slices((augmented_files,specaugment_flags,augmented_secs, augmented_labels))
    train_ds = train_ds.map(Dp.map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(1000).batch(8).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_files, val_labels))
    val_ds = val_ds.map(lambda fi, la: Dp.tf_wrapper(fi,0.0, la, training=False), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(8).prefetch(tf.data.AUTOTUNE)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(cfg.N_MFCC, cfg.expected_frames, 1)),

        # 1.CONV
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.3),

        # 2.CONV
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.3),

        # 3.CONV
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.4),

        #GAP
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(len(class_labels), activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',   # metric to watch
        factor=0.5,          # LR multiplier (reduce by half)
        patience=3,          # wait 3 epochs with no improvement
        verbose=1,           # print messages when LR changes
        min_lr=1e-6          # minimum LR allowed
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',     # metric to watch, e.g. 'val_loss' or 'val_accuracy'
        patience=5,             # number of epochs to wait before stopping
        verbose=1,              # print message when stopping
        restore_best_weights=True  # restore model weights from the epoch with the best value
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30 ,
        callbacks=[lr_callback, early_stopping]
    )

    scores = model.evaluate(val_ds)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]}={scores[0]}, {model.metrics_names[1]}={scores[1] * 100:.2f}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_filename = f"model_fold{fold_no}_{timestamp}.h5"
    my_model_path = os.path.join(cfg.MODELS_PATH, model_filename)
    model.save(my_model_path)
    print(f"Saved model for fold {fold_no} to {my_model_path}")
    fold_no += 1
