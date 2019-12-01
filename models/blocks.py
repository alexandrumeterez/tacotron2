import torch.nn as nn


class ConvBlock(nn.Module):
    """
        Block containing Conv1d, BatchNorm, Dropout and Relu/Tanh activation
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding, activation='relu'):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                padding=padding)
        self.batchnorm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout()
        self.activation = None
        if activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Tanh()

    def forward(self, input):
        out = self.conv1d(input)
        out = self.batchnorm(out)
        out = self.dropout(out)
        out = self.activation(out)
        return out


class PreNet(nn.Module):
    """
        From the paper:
            pre-net containing 2 fully connected layers
            of 256 hidden ReLU units
        Input size: 80, equal to the number of Mel filters in the
        filterbank
    """

    def __init__(self, in_channels=80, out_channels=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU()
        )
        self.dropout = nn.Dropout()

    def forward(self, input):
        out = self.fc(input)
        out = self.dropout(out)
        return out


class PostNet(nn.Module):
    """
        From the paper: Finally, the predicted mel spectrogram is passed
            through a 5-layer convolutional post-net which predicts a residual
            to add to the prediction to improve the overall reconstruction. Each
            post-net layer is comprised of 512 filters with shape 5 × 1 with batch
            normalization, followed by tanh activations on all but the final layer
    """

    def __init__(self):
        super().__init__()
        self.conv1 = ConvBlock(in_channels=1, out_channels=512, kernel_size=5, padding=2, activation='tanh')
        self.conv2 = ConvBlock(in_channels=512, out_channels=512, kernel_size=5, padding=2, activation='tanh')
        self.conv3 = ConvBlock(in_channels=512, out_channels=512, kernel_size=5, padding=2, activation='tanh')
        self.conv4 = ConvBlock(in_channels=512, out_channels=512, kernel_size=5, padding=2, activation='tanh')

        self.conv5 = nn.Conv1d(in_channels=512, out_channels=1, kernel_size=5, padding=2)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        return out


class Encoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        """
            From the paper:
                Input characters are represented using a learned 512-dimensional
                character embedding, which are passed through a stack of 3 convolutional layers each containing 512 filters with shape 5 × 1, i.e., where
                each filter spans 5 characters, followed by batch normalization [18]
                and ReLU activations.
            Padding is added such that the output shape is the same as the input.
            See: https://pytorch.org/docs/stable/nn.html#torch.nn.Conv1d for the formula
        """
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=512, padding_idx=0)
        self.conv1 = ConvBlock(in_channels=vocab_size, out_channels=vocab_size, kernel_size=5, padding=2)
        self.conv2 = ConvBlock(in_channels=vocab_size, out_channels=vocab_size, kernel_size=5, padding=2)
        self.conv3 = ConvBlock(in_channels=vocab_size, out_channels=vocab_size, kernel_size=5, padding=2)
        self.rnn = nn.LSTM(input_size=vocab_size, hidden_size=256, bidirectional=True, dropout=0.1)

    def forward(self, input):
        # input
        # (batch_size, seq_len)

        out = self.embedding(input)
        # (batch_size, seq_len, vocab_size)

        out = out.permute(0, 2, 1)
        # (batch_size, vocab_size, seq_len)

        out = self.conv1(out)
        # (batch_size, vocab_size, seq_len)

        out = self.conv2(out)
        # (batch_size, vocab_size, seq_len)

        out = self.conv3(out)
        # (batch_size, vocab_size, seq_len)

        out = out.permute(2, 0, 1)
        # (seq_len, batch_size, vocab_size)

        output, _ = self.rnn(out)
        # output of shape (seq_len, batch, num_directions * hidden_size):
        # [forward1, backward1, forward2, backward2 etc]
        # (seq_len, batch_size, 2 * 256)

        # For the unpacked case, the directions can be separated using output.view(seq_len, batch, num_directions, hidden_size), with forward and backward being direction 0 and 1 respectively.
        output_directions = output.view(output.shape[0], output.shape[1], 2, 256)
        output = output_directions[:, :, 0, :] + output_directions[:, :, 1, :]

        return output

