from datasets.speechdataset import LJSpeechDataset
from utils.vocab import Vocabulary

ds = LJSpeechDataset('data/LJSpeech-1.1')
v = Vocabulary()
all_strings = [x[0] for x in ds]
v.build_vocab(all_strings)
