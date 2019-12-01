from datasets.speechdataset import LJSpeechDataset
from utils.utils import show_spectrogram

ds = LJSpeechDataset('data/LJSpeech-1.1')
text, spectrogram = ds[0]
print(spectrogram.shape)
text, spectrogram = ds[10]
print(spectrogram.shape)
