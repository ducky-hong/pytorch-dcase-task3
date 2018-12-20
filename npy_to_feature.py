import os
import glob
import numpy as np
from tqdm import *
from librosa.feature import melspectrogram

def log_mel(sample, sample_rate, frame_size=40, frame_step=20,
            n_mels=64, fmin=0.0, fmax=None):
    return np.log(melspectrogram(sample.astype(np.float32), sample_rate,
                                 n_fft=int(sample_rate * frame_size / 1000),
                                 hop_length=int(sample_rate * frame_step / 1000),
                                 n_mels=n_mels, fmin=fmin, fmax=fmax,
                                 power=1.0) + 1e-8).T

def main():
    means = [4.987868109063186, 5.600202438886607, 4.8412715881907]
    stds = [1.58625372, 1.60122786, 3.05533208]
    sample_rates = [8000, 22050, 44100]
    for i, sample_rate in enumerate(sample_rates):
        for filename in tqdm(glob.glob('./datasets/npy/{}/*/*/*.npy'.format(sample_rate))):
            data = np.load(filename)
            data = log_mel(data, sample_rate, fmin=2000)
            data = (data - means[i]) / stds[i]
            feature_filename = filename.replace('npy', 'feature_npy', 1)
            os.makedirs(os.path.dirname(feature_filename), exist_ok=True)
            np.save(feature_filename, data)

main()

