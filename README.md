# 🎙️ MLAudioController

**Generate keyboard inputs based on speech commands** using a CNN trained on MFCC audio features.  
Speak → Model listens → Corresponding key press is triggered.

---

## 🖥️ Environment

- **Python version:** 3.12.11  
- Install dependencies with:

```bash
pip install -r requirements.txt
```

## 🗣️ Available Speech Commands (Controller Buttons)

(Model is trained on English pronunciation)

| Class | Command | Mapped Key (default) |
| ----- | ------- |----------------------|
| 0     | a       | —                    |
| 1     | b       | —                    |
| 2     | down    | ⬇️                   |
| 3     | home    | —                    |
| 4     | l1      | —                    |
| 5     | l2      | —                    |
| 6     | l3      | —                    |
| 7     | left    | ⬅️                   |
| 8     | r1      | —                    |
| 9     | r2      | —                    |
| 10    | r3      | —                    |
| 11    | right   | ➡️                   |
| 12    | select  | —                    |
| 13    | start   | --                   |
| 14    | up      | ⬆️                   |
| 15    | x       | —                    |
| 16    | y       | —                    |

## 🚀 How to Use

1.  Clone the repository
2. Open CONFIG.py and adjust settings if needed. 
3. Prepare data: Create a data folder or unpack the provided data.zip.
4. Run main.py
5. Start recording:
    
    Press your defined RECORD_BUTTON (default: F8).

    Speak your command.

    Model classifies audio and simulates key press.

⏳ Note: The first prediction is slower due to model loading.

## ⚙️ Configuration Highlights

CONFIG.py contains all adjustable settings, but the most important ones are:

### Key bindings
Map each class to a specific key in CLASS_TO_KEY.

Recording trigger key
Default: F8. Change if it conflicts with your application.

### Model choice

Multiple pre-trained models available in /models.

Choose between 16kHz or 48kHz versions by setting modelname.

48kHz models generally gave me better results but takes 2–3× longer to train.

(Every pretrained version has its weakpoints. If one doesnt work for you try a different one, but none are perfect because of a lack of training data.)

### Clip length
Adjust AUDIO_LENGTH_SECONDS for longer/shorter training and recording clips.

Sampling rate (SR)
Must match training sampling rate for correct classification.

## 🏋️ Training Your Own Model

1. Create a data/ folder in the project root.

2. Organize your samples:

    One folder per class label.

    Place .wav files inside their respective folders.

3.  Adjust data augmentation parameters in CONFIG.py (defaults are aggressive).

    Set your desired SamplingRate (16kHz or 48kHz).

4.  Run MODEL.py (Per execution the standard saves 5 models to /models)

## 🐞 Known Issues

If the model doesn’t respond or classify correctly:

1.  Ensure your microphone is working.

2. Check that it’s set as your default recording device.

    The sounddevice library uses your OS’s default input device.
