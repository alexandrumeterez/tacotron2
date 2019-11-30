from torch.utils.data import Dataset
import pandas as pd
import os
import librosa
from config.hyperparameters import *
from tqdm import tqdm
import numpy as np


def load_wav(filename):
    return librosa.load(filename, sr=SAMPLE_RATE)


def ms_to_frames(ms, sample_rate):
    return int((ms / 1000) * sample_rate)


def wav_to_spectrogram(wav, sample_rate=SAMPLE_RATE,
                       fft_frame_size=FFT_FRAME_SIZE,
                       fft_hop_size=FFT_HOP_SIZE,
                       num_mels=NUM_MELS,
                       min_freq=MIN_FREQ,
                       max_freq=MAX_FREQ,
                       floor_freq=FLOOR_FREQ):
    n_fft = ms_to_frames(fft_frame_size, sample_rate)
    hop_length = ms_to_frames(fft_hop_size, sample_rate)
    mel_spec = librosa.feature.melspectrogram(wav, sr=sample_rate,
                                              n_fft=n_fft,
                                              hop_length=hop_length,
                                              n_mels=num_mels,
                                              fmin=min_freq,
                                              fmax=max_freq)
    return librosa.power_to_db(mel_spec, ref=floor_freq).T


class LJSpeechDataset(Dataset):
    def __init__(self, path, force_make_spectrograms=False):
        self.path = path
        file = os.path.join(path, 'metadata.csv')
        self.metadata = pd.read_csv(file, sep='|', names=['wav', 'short', 'text'], usecols=[0, 2]).dropna()
        self.metadata['length'] = self.metadata['wav'].apply(
            lambda x: librosa.get_duration(filename=os.path.join(path, 'wavs', '{}.wav'.format(x))))
        if force_make_spectrograms or not os.path.exists(os.path.join(path, 'spectrograms')):
            os.mkdir(os.path.join(path, 'spectrograms'))
            self.make_spectrograms()

    def make_spectrograms(self):
        wavs = self.metadata['wav']
        for wav in tqdm(wavs):
            wav_file = os.path.join(self.path, 'wavs', wav + '.wav')
            audio, _ = load_wav(wav_file)
            spectrogram = wav_to_spectrogram(audio)
            np.save(os.path.join(self.path, 'spectrograms', wav + '.npy'), spectrogram)

    def __getitem__(self, item):
        text = self.metadata.iloc[item]['text']
        filename = self.metadata.iloc[item]['wav']

        spectrogram = np.load(os.path.join(self.path, 'spectrograms', filename + '.npy'))

        return text, spectrogram

    def __len__(self):
        return len(self.metadata)
