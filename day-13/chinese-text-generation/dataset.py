import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, path, seq_length=50):
        super(TextDataset).__init__()
        self.path = path
        self.seq_length = seq_length

        self.parse_corpus()

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return (self.X[i], self.Y[i])

    def parse_corpus(self):
        '''
        Parse raw corpus text into input-output tensor pairs
        '''
        # Read text from file
        with open(self.path, 'r') as f:
            raw_text = f.read().replace('\n', '')

        # Get unique characters
        chars = sorted(list(set(raw_text)))

        # Map char to int / int to char
        char_to_int = dict((c, i) for i, c in enumerate(chars))
        int_to_char = dict((i, c) for i, c in enumerate(chars))

        # Prepare training data, for every <seq_length> chars, predict 1 char after the sequence
        n_chars = len(raw_text)
        X = [] # N x self.seq_length
        Y = [] # N x self.seq_length
        for i in range(0, n_chars - self.seq_length, self.seq_length):
            seq_in = raw_text[i:i+self.seq_length]
            seq_out = raw_text[i+1:i+1+self.seq_length]
            X.append([char_to_int[char] for char in seq_in])
            Y.append([char_to_int[char] for char in seq_out])

        # Convert to tensor
        X = torch.tensor(X, dtype=torch.long)
        Y = torch.tensor(Y, dtype=torch.long)

        self.char_to_int, self.int_to_char, self.chars = char_to_int, int_to_char, chars
        self.X, self.Y = X, Y
        self.n = len(X)
