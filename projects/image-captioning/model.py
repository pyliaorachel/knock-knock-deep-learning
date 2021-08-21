import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ImageEncoder(nn.Module):
    def __init__(self, device, hidden_dim=1024, dropout=0.2):
        super(ImageEncoder, self).__init__()

        self.device = device

        self.conv1 = nn.Conv2d(3, 64, 3)
        self.max_pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.max_pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.max_pool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(256, 512, 3)
        self.max_pool4 = nn.MaxPool2d(2)
        self.conv5 = nn.Conv2d(512, 1024, 3)
        self.max_pool5 = nn.MaxPool2d(2)

        self.dropout = nn.Dropout2d(dropout)
        self.fc = nn.Linear(200704, hidden_dim)

    def forward(self, imgs):
        y = self.max_pool1(torch.relu(self.conv1(imgs)))
        y = self.dropout(y)
        y = self.max_pool2(torch.relu(self.conv2(y)))
        y = self.dropout(y)
        y = self.max_pool3(torch.relu(self.conv3(y)))
        y = self.dropout(y)
        y = self.max_pool4(torch.relu(self.conv4(y)))
        y = self.dropout(y)
        y = self.max_pool5(torch.relu(self.conv5(y)))
        y = torch.flatten(y, 1)
        y = self.fc(y)
        return y

class CaptionDecoder(nn.Module):
    def __init__(self, device, n_vocab, enc_hidden_dim=1024, embedding_dim=256, dec_hidden_dim=256, dropout=0.2):
        super(CaptionDecoder, self).__init__()

        self.device = device
        self.n_vocab = n_vocab

        self.enc_to_dec_h = nn.Linear(enc_hidden_dim, dec_hidden_dim)
        self.enc_to_dec_c = nn.Linear(enc_hidden_dim, dec_hidden_dim)

        self.emb = nn.Embedding(n_vocab, embedding_dim)
        self.lstm_cell = nn.LSTMCell(embedding_dim, dec_hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(dec_hidden_dim, n_vocab)

    def decode_to_end(self, img_embedding, n_vocab, start_token_idx, end_token_idx, max_seq_len=50):
        last_token = start_token_idx
        caption = [last_token]
        h = img_embedding
        c = img_embedding
        while len(caption) < max_seq_len and last_token != end_token_idx: 
            # Construct input
            word = torch.tensor([last_token], dtype=torch.long).to(self.device)
            word_embedding = self.emb(word)

            # Predict next character
            with torch.no_grad():
                h, c = self.lstm_cell(word_embedding, (h, c))
                pred = self.fc(h)
                prob = F.softmax(pred, dim=1)[0]

            # Pick word based on probability instead of always picking the highest value
            word_idx = np.random.choice(list(range(n_vocab)), p=prob.numpy())

            caption.append(word_idx)
            last_token = word_idx
        return caption

    def forward(self, img_embeddings, captions, caption_lengths):
        batch_size = img_embeddings.size(0)

        # Sort captions by decreasing lengths, and create embeddings
        caption_lengths, sort_idx = caption_lengths.sort(descending=True)
        img_embeddings = img_embeddings[sort_idx]
        captions = captions[sort_idx]
        caption_embeddings = self.emb(captions)

        # Minus 1 in length to avoid decoding at <end> token
        caption_lengths = (caption_lengths - 1).tolist()
        max_caption_length = max(caption_lengths)

        # Init hidden states
        h = self.enc_to_dec_h(img_embeddings)
        c = self.enc_to_dec_c(img_embeddings)

        # Init prediction as tensor
        predictions = torch.zeros(batch_size, max_caption_length, self.n_vocab).to(self.device)

        # Iterate and output each timestep of the captions
        for t in range(max_caption_length):
            # batch size at this timestep
            batch_size_t = sum([l > t for l in caption_lengths])

            # don't decode for captions who already ended
            x = caption_embeddings[:batch_size_t, t]
            h = h[:batch_size_t]
            c = c[:batch_size_t]

            h, c = self.lstm_cell(x, (h, c))
            predictions[:batch_size_t, t] = self.fc(self.dropout(h))

        return predictions, captions, caption_lengths, sort_idx
