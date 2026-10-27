import os
from pathlib import Path

import sckit
import numpy as np
from scipy.io import wavfile

# The part below is meant as a personal todo list for the design of the class as i can't just input the wav data into a model without some pre-processing.
# was kinda just easier to place it in the file for developing rather than a seperate one at the time really.

# Create a class that performs the following:
#   + Loads in wavfiles as numpy arrays based of the base dir
#   + creates mfcc's for each data set from the wavfiles
#   + able to differenciate between real and fake datasets.
#   + able to print out the datasets (from real or fake)


class AudioPreProcessor:
    def __init__(self):
        self.load_path = ""
        self.real_nps = []
        self.real_mfcc = []

        self.fake_nps = []
        self.fake_mfcc = []


    # these load methods might be changed in the future due to their repeatitiveness but rn they do work as intended
    def load_fake(self):
        read_path = str(self.load_path / 'fake')
        for fname in os.listdir(read_path):
            wav_fname = os.path.join(read_path, fname)
            if os.path.isfile(wav_fname):
                self.real_nps.append(wavfile.read(wav_fname))


    def load_real(self):
        read_path = str(self.load_path / 'real')
        for fname in os.listdir(read_path):
            wav_fname = os.path.join(read_path, fname)
            if os.path.isfile(wav_fname):
                self.real_nps.append(wavfile.read(wav_fname))

    def load(self):
        # TODO: need to include an option for specifying training and validation for the load path.
        self.load_path = Path(os.getcwd()) / 'for-norm' / 'training'
        self.load_fake()
        self.load_real()

    def convert_to_mfcc(self):
        for nps in self.real_nps:
            self.real_mfcc.append(librosa.feature.mfcc(y=nps))

        for nps in self.fake_nps:
            self.fake_mfcc.append(librosa.feature.mfcc(y=nps))


    def show_real_mfcc(self):
        return self.real_mfcc

    def show_fake_mfcc(self):
        return self.fake_mfcc

processor = AudioPreProcessor()
processor.load()
