import os
from pathlib import Path

import librosa
import sckit
import numpy as np
from scipy.io import wavfile as wf

# The part below is meant as a personal todo list for the design of the class as i can't just input the wav data into a model without some pre-processing.
# was kinda just easier to place it in the file for developing rather than a seperate one at the time really.

# Create a class that performs the following:
#   + Loads in wavfiles as numpy arrays based of the base dir
#   + creates mfcc's for each data set from the wavfiles
#   + able to differenciate between real and fake datasets.
#   + able to print out the datasets (from real or fake)


class AudioPreProcessor:
    def __init__(self):
        # Forgot to manually label the datasets so that this can pass through a supervised algorithm
        self.total_nps = []

        self.labels = []
        self.mfccs = []


    def load(self, path):
        # TODO: include error detection for when the path is invalid.
        self.load_path = Path(os.getcwd()) / 'for-norm' / path
        # Label the name of the directories: real and fake.
        categories = ['real', 'fake']
        for category in categories:
            # Builds a path to read the wav files from.
            read_path = str(self.load_path / category)
            # Looks for file names under the path and if's its a file then load it in through
            # librosa load method.
            for fname in os.listdir(read_path):
                wav_fname = os.path.join(read_path, fname)
                if os.path.isfile(wav_fname):
                    # Loading the wav data whilst preserving the sample rate of said file
                    wav_data, sr = librosa.load(wav_fname, sr=None)
                    #TODO: perform some assertions here to ensure that data is correct.
                    self.total_nps.append([wav_data, sr, category])

    def convert_to_mfcc(self):
        for nps in self.total_nps:
            y, sr, label = nps
            self.mfccs.append(librosa.feature.mfcc(y=y, sr=sr))
            self.labels.append(label)

    def clear(self):
        self.total_nps = []
        self.labels = []
        self.mfccs = []

    def show_mfccs(self):
        return self.mfccs

    def show_labels(self):
        return self.labels
