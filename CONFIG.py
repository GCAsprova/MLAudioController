import os
from pynput import keyboard
#CONFIG FILE -- Configure all variables for Training and Audioparameters here

#Keybinds
RECORD_BUTTON = keyboard.Key.f8

#For letters and basic symbols just put them as characters to the corresponding class id
#For special keys like shift, space, F1 ... F12 use keyboard.Key.shift( or space or ctrl or ...)

CLASS_TO_KEY = {
    0: '', # AudioCommand = a
    1: '', # AudioCommand = b
    2: keyboard.Key.down, # AudioCommand = down
    3: '', # AudioCommand = home
    4: '', # AudioCommand = l1
    5: '', # AudioCommand = l2
    6: '', # AudioCommand = l3
    7: keyboard.Key.left, # AudioCommand = left
    8: '', # AudioCommand = r1
    9: '', # AudioCommand = r2
    10: '',# AudioCommand = r3
    11: keyboard.Key.right,# AudioCommand = right
    12: '',# AudioCommand = select
    13: '',# AudioCommand = start
    14: keyboard.Key.up,# AudioCommand = up
    15: '',# AudioCommand = x
    16: '',# AudioCommand = y
}

#General
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) #Base Directory
MODELS_PATH = os.path.join(BASE_DIR, 'models') #Models folder
DATA_PATH = os.path.join(BASE_DIR, 'data') #Data folder


#Model
modelname = "model_v2_48khz_fold2.h5" #Saved model weights to use during execution
SHIFT_TIMES = [0.0,+0.2,-0.2,0.1,-0.1,+0.3,-0.3] #How many and what timeshifts should be applied to each base sample during training
num_augmentations = 5 #How many Specaugmentations per TimeShifted Sample
k = 5 # Number of folds for KFoldCrossValidation



#Audio specifications
SR = 48000  # Audio sampling rate in Hz
N_MFCC = 40  # Number of MFCC (Mel-Frequency Cepstral Coefficients) features to extract per frame

MAX_MASK_PCT = 0.1  # Maximum percentage of the spectrogram that can be masked for data augmentation
N_FREQ_MASKS = 1    # Number of frequency masks to apply during augmentation
N_TIME_MASKS = 1    # Number of time masks to apply during augmentation

AUDIO_LENGTH_SECONDS = 1.5  # Target fixed audio length in seconds
AUDIO_LENGTH = int(SR * AUDIO_LENGTH_SECONDS)  # Target fixed audio length in samples

hop_length = int(SR * 0.010)  # Frame hop length in samples (10 ms step size)
n_fft = int(SR * 0.025)       # FFT window size in samples (25 ms analysis window)

expected_frames = 1 + int((AUDIO_LENGTH - n_fft) // hop_length)
# Expected number of feature frames for an audio clip of fixed length

