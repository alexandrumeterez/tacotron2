from models.blocks import *
import torch.nn as nn
from torch.autograd import Variable
import random


class Tacotron2(nn.Module):
    def __init__(self, vocab, device, teacher_forcing_ratio=1.0):
        super().__init__()
        self.vocab = vocab
        self.encoder = Encoder(self.vocab.n_chars)
        self.decoder = Decoder(device)
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(self, text, targets=None):
        encoder_out = self.encoder(text)

        # daca targets nu e None, atunci e in training si folosesc teacher forcing
        # altfel, e in test
        use_teacher_forcing = (targets is not None)

        maxlen = len(targets)

        seq1_len, batch_size, _ = encoder_out.size()
        outputs = Variable(encoder_out.data.new(maxlen, batch_size, 80))
        stop_tokens = Variable(outputs.data.new(maxlen, batch_size))
        masks = torch.zeros(maxlen, batch_size, seq1_len)

        output = Variable(outputs.data.new(1, batch_size, 80).fill_(0))
        mask = self.decoder.init_mask(encoder_out)
        hidden = self.decoder.init_hidden(batch_size)

        for t in range(maxlen):
            output, stop_token, hidden, mask = self.decoder(output, encoder_out, hidden, mask)
            outputs[t] = output
            stop_tokens[t] = stop_token.squeeze()
            masks[t] = mask.data
            if use_teacher_forcing and random.random() < self.teacher_forcing_ratio:
                output = targets[t].unsqueeze(0)
        return outputs, stop_tokens.transpose(1, 0), masks.permute(1, 2, 0)  # batch, src, trg
