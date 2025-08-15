import os

#CONFIG FILE -- Configure all variables for Training and Audioparameters here

#General
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) #Base Directory
MODELS_PATH = os.path.join(BASE_DIR, 'models') #Models folder
DATA_PATH = os.path.join(BASE_DIR, 'data') #Data folder


#Model
modelname = "model_v1_48khz.h5" #Modelweights to use during execution
SHIFT_TIMES = [0.0,+0.2,-0.2,0.1,-0.1,+0.3,-0.3] #How many and what timeshifts should be applied to each base sample
num_augmentations = 5 #How many Specaugmentations per TimeShifted Sample
k = 5 # Number of folds for KFoldCrossValidation



#Audio specifications
SR = 48000  # Audio sampling rate in Hz (16 kHz is common for speech recognition)
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

