import os
from pathlib import Path

import librosa
import sckit
import numpy as np
import soundfile as sf
import gc

# The part below is meant as a personal todo list for the design of the class as i can't just input the wav data into a model without some pre-processing.
# was kinda just easier to place it in the file for developing rather than a seperate one at the time really.

# Create a class that performs the following:
#   + Loads in wavfiles as numpy arrays based of the base dir
#   + creates mfcc's for each data set from the wavfiles
#   + able to differenciate between real and fake datasets.
#   + able to print out the datasets (from real or fake)

# This is for padding out the numpy arrays as they're not homogenous in length (current fix, might get back to this at some point)
MAX_LEN = 20 * 400


class AudioPreProcessor:
    def __init__(self):
        # Forgot to manually label the datasets so that this can pass through a supervised algorithm
        self.labels = []
        self.mfccs = []


    def load(self, path):
        # TODO: include error detection for when the path is invalid.
        self.load_path = Path(os.getcwd()) / 'for-norm' / path
        # Label the name of the directories: real and fake.
        categories = ['real', 'fake']
        for category in categories:
            # Builds a path to read the wav files from.
            read_path = self.load_path / category
            # Finds all wav files in the directory with a wildcard
            for wav_fname in read_path.glob('*.wav'):
                # soundfile is used as it's deemed to be a fast wav load library
                wav_data, sr = sf.read(str(wav_fname))
                mfcc = librosa.feature.mfcc(y=wav_data, sr=sr)
                mfcc = mfcc.flatten()
                print(mfcc.shape)
                len_diff = MAX_LEN - len(mfcc)
                # This is to pad out the array so that their all homogenous.
                if len_diff > 0:
                    mfcc = np.pad(mfcc, (0, len_diff), 'mean')
                print(mfcc.shape)
                self.mfccs.append(mfcc)
                self.labels.append(category)
                del mfcc
                del wav_data
            gc.collect()
            print(f"Finished loading the {category} wav files for {path}")

    def clear(self):
        self.labels = []
        self.mfccs = []
        gc.collect()

    def show_mfccs(self):
        return self.mfccs

    def show_labels(self):
        return self.labels
