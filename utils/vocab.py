class Vocabulary:
    def __init__(self):
        self.char_to_id = {'<pad>': 0, '<unk>': 1, '<eos>': 2}
        self.id_to_char = {0: '<pad>', '<unk>': 1, '<eos>': 2}
        self.n_chars = 3

    def build_fake_vocab(self):
        chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'\"(),-.:;? ';
        for c in chars:
            self.char_to_id[c] = self.n_chars
            self.id_to_char[self.n_chars] = c
            self.n_chars += 1

    def build_vocab(self, sentences):
        long_string = ''.join(sentences)
        for c in long_string:
            if c not in self.char_to_id:
                self.char_to_id[c] = self.n_chars
                self.id_to_char[self.n_chars] = c
                self.n_chars += 1

    def text2seq(self, text):
        seq = [self.char_to_id.get(c, self.char_to_id['<unk>']) for c in text]
        return seq + [2]

    def seq2text(self, seq):
        return "".join(self.id_to_char.get(i, '<unk>') for i in seq)
