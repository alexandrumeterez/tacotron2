from datasets.speechdataset import LJSpeechDataset
from utils.vocab import Vocabulary
import torch
import numpy as np
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from models.tacotron2 import Tacotron2
import librosa
import librosa.display
import matplotlib.pyplot as plt


def collate_fn(batch):
    text = [item[0] for item in batch]
    audio = [item[1] for item in batch]

    text_lengths = [len(x) for x in text]
    audio_lengths = [len(x) for x in audio]

    max_text = max(text_lengths)
    max_audio = max(audio_lengths)

    text_tostack = [pad1d(x, max_text) for x in text]
    audio_tostack = [pad2d(x, max_audio) for x in audio]

    text_batch = np.stack(text_tostack)
    audio_batch = np.stack(audio_tostack)

    return (torch.LongTensor(text_batch),
            torch.FloatTensor(audio_batch).permute(1, 0, 2),
            text_lengths, audio_lengths)


def pad1d(seq, max_len):
    return np.pad(seq, (0, max_len - len(seq)), mode='constant', constant_values=0)


def pad2d(seq, max_len, dim=80, pad_value=0.0):
    padded = np.zeros((max_len, dim)) + pad_value
    padded[:len(seq), :] = seq
    return padded


def train(model, optimizer, dataset, batch_size, device):
    model.train()
    loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=1, batch_size=batch_size)
    pbar = tqdm(loader, total=len(loader), unit=' batches')
    curr_loss = 0.0
    for idx, (text, audio, text_lengths, audio_lengths) in enumerate(pbar):
        print(idx)
        text = Variable(text).to(device)
        targets = Variable(audio, requires_grad=False).to(device)
        outputs, attn = model(text, targets)
        spec_loss = F.mse_loss(outputs, targets)
        loss = spec_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        curr_loss += loss.item()
    print('Loss: {}'.format(curr_loss))


def evaluate(model, text, vocab):
    model.eval()
    text = torch.LongTensor(vocab.text2seq(text)).unsqueeze(0)
    outputs, attn = model(text, None)

    return outputs.squeeze(), attn


def get_waveform(model, text, vocab):
    outputs, attn = evaluate(model, text, vocab)
    outputs = outputs.detach().numpy()
    S = librosa.core.db_to_power(outputs)
    S_ = librosa.feature.inverse.mel_to_stft(S, n_fft=1024, power=0.2)
    librosa.display.specshow(S_, fmax=8000)
    plt.show()
    y = librosa.griffinlim(S_, 50)
    return y


def main():
    dataset = LJSpeechDataset('data/LJSpeech-1.1')
    # sentences = [x[0] for x in dataset]
    vocab = Vocabulary()
    # vocab.build_vocab(sentences)

    vocab.build_fake_vocab()

    dataset.vocab = vocab
    dataset.convert_text2seq()
    DEVICE = 'cpu'
    model = Tacotron2(vocab, device=DEVICE, teacher_forcing_ratio=0.7)
    model.to(DEVICE)
    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-6,
                     betas=(0.9, 0.999), eps=1e-6)
    BATCH_SIZE = 1
    # train(model, optimizer, dataset, BATCH_SIZE, DEVICE)
    # outputs, attn = evaluate(model, "I like apples.", vocab)
    y = get_waveform(model, "I like apples.", vocab)


if __name__ == '__main__':
    main()
