import torch.nn as nn
import torch.nn.functional as F
import torch


# TODO: UNDERSTAND!!!!!!
class LocationAttention(nn.Module):
    """
    Calculates context vector based on previous decoder hidden state (query vector),
    encoder output features, and convolutional features extracted from previous attention weights.
    Attention-Based Models for Speech Recognition
    https://arxiv.org/pdf/1506.07503.pdf
    """

    def __init__(self, encoded_dim, query_dim, attention_dim, num_location_features=32):
        super(LocationAttention, self).__init__()
        self.f = nn.Conv1d(in_channels=1, out_channels=num_location_features,
                           kernel_size=31, padding=15, bias=False)
        self.U = nn.Linear(num_location_features, attention_dim)
        self.W = nn.Linear(query_dim, attention_dim)
        self.V = nn.Linear(encoded_dim, attention_dim)
        self.w = nn.Linear(attention_dim, 1, bias=False)
        self.tanh = nn.Tanh()

    def score(self, query_vector, encoder_out, mask):
        encoder_out = self.V(encoder_out)  # (seq, batch, atten_dim) # project to attn dim
        query_vector = self.W(query_vector)  # (seq, batch, atten_dim)
        attention_energies = encoder_out + query_vector
        location_features = self.f(mask.permute(1, 0, 2))  # (batch, 1, seq1_len)
        attention_energies += self.U(location_features.permute(2, 0, 1))  # (seq, batch, numfeats)
        return self.w(self.tanh(attention_energies))

    def forward(self, query_vector, encoder_out, mask):
        energies = self.score(query_vector, encoder_out, mask)
        mask = F.softmax(energies, dim=0)
        context = encoder_out.permute(1, 2, 0) @ mask.permute(1, 0, 2)  # (batch, seq1, seq2)
        context = context.permute(2, 0, 1)  # (seq2, batch, encoder_dim)
        mask = mask.permute(2, 1, 0)  # (seq2, batch, seq1)
        return context, mask


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

        """
            From the paper:
                The output of the final convolutional layer is passed into a
                single bi-directional [19] LSTM [20] layer containing 512 units (256
                in each direction) to generate the encoded features.
        """

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


class Decoder(nn.Module):
    """
        The decoder is an autoregressive recurrent neural network which
        predicts a mel spectrogram from the encoded input sequence one
        frame at a time.
    """

    def __init__(self):
        super().__init__()
        """
        The prediction from the previous time step is first
        passed through a small pre-net containing 2 fully connected layers
        of 256 hidden ReLU units. 
        """
        self.prenet = PreNet(in_channels=80, out_channels=256)
        """
        The encoder output is consumed by an attention network which
        summarizes the full encoded sequence as a fixed-length context vector
        for each decoder output step.
        """
        self.attention = LocationAttention(encoded_dim=256, query_dim=1024, attention_dim=128)
        """
        The prenet output and attention context vector are concatenated and passed
        through a stack of 2 uni-directional LSTM layers with 1024 units.
        """
        # x2 because of the concatenation
        self.rnn = nn.LSTM(input_size=256 * 2, hidden_size=1024, num_layers=2, dropout=0.1)
        """
        The concatenation of the LSTM output and the attention context
        vector is projected through a linear transform to predict the target
        spectrogram frame.
        """
        self.spec_out = nn.Linear(in_features=1024 + 256, out_features=80)
        """
        Finally, the predicted mel spectrogram is passed
        through a 5-layer convolutional post-net which predicts a residual
        to add to the prediction to improve the overall reconstruction
        """
        self.postnet = PostNet()
        """
        In parallel to spectrogram frame prediction, the concatenation of
        decoder LSTM output and the attention context is projected down
        to a scalar and passed through a sigmoid activation to predict the
        probability that the output sequence has completed.
        """
        self.stop_out = nn.Linear(in_features=1024 + 256, out_features=1)

    def forward(self, previous_out, encoder_out, decoder_hidden=None, mask=None):
        """
        Decodes a single frame
        """
        previous_out = self.prenet(previous_out)  # (4, 1, 256)
        hidden, cell = decoder_hidden
        context, mask = self.attention(hidden[:-1], encoder_out, mask)
        rnn_input = torch.cat([previous_out, context], dim=2)
        rnn_out, decoder_hidden = self.rnn(rnn_input, decoder_hidden)
        spec_frame = self.spec_out(torch.cat([rnn_out, context], dim=2))  # predict next audio frame
        stop_token = self.stop_out(torch.cat([rnn_out, context], dim=2))  # predict stop token
        spec_frame = spec_frame.permute(1, 0, 2)
        spec_frame = spec_frame + self.postnet(spec_frame)  # add residual
        return spec_frame.permute(1, 0, 2), stop_token, decoder_hidden, mask
