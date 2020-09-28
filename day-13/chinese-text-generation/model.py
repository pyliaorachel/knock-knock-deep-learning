import math

import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, n_vocab, embedding_dim, hidden_dim, nhead=8, num_layers=6, dropout=0.2):
        super(Net, self).__init__()

        self.src_mask = None

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.embeddings = nn.Embedding(n_vocab, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, nhead, dim_feedforward=hidden_dim, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.hidden2out = nn.Linear(embedding_dim, n_vocab)

    def _generate_square_subsequent_mask(self, sz):
        # Each row i in mask will have column [0, i] set to 1, [i+1, sz) set to -inf
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        src = src.t()

        # For each input subsequence, create a mask to mask out future sequences
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            mask = self._generate_square_subsequent_mask(len(src))
            self.src_mask = mask

        embeddings = self.embeddings(src) # seq_len x batch_size x embed_dim
        x = self.pos_encoder(embeddings)
        out = self.transformer_encoder(x, self.src_mask)
        out = self.hidden2out(out)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
