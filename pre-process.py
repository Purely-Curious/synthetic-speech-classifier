import os
from pathlib import Path

import sckit
import numpy as np
from scipy.io import wavfile

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
        self.load_path = Path(os.getcwd()) / 'for-norm' / 'training'
        self.load_fake()
        #self.load_real()

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
print(processor.load_path)
