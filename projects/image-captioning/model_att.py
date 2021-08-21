import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ImageEncoder(nn.Module):
    def __init__(self, device, dropout=0.2):
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
        return y

class CaptionDecoder(nn.Module):
    def __init__(self, device, n_vocab, embedding_dim=256, enc_hidden_dim=1024, dec_hidden_dim=256, dropout=0.2):
        super(CaptionDecoder, self).__init__()

        self.device = device
        self.n_vocab = n_vocab
        self.embedding_dim = embedding_dim
        self.enc_hidden_dim = enc_hidden_dim
        self.dec_hidden_dim = dec_hidden_dim

        self.enc_to_dec_h = nn.Linear(enc_hidden_dim, dec_hidden_dim)
        self.enc_to_dec_c = nn.Linear(enc_hidden_dim, dec_hidden_dim)

        self.emb = nn.Embedding(n_vocab, embedding_dim)
        self.lstm_cell = nn.LSTMCell(embedding_dim + enc_hidden_dim, dec_hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(dec_hidden_dim, n_vocab)

        self.attention = Attention(enc_hidden_dim, dec_hidden_dim)

    def init_hidden_cell_states(self, img_embeddings):
        # Init hidden and cell states by taking mean of img embedding over all pixels
        mean_img_embeddings = img_embeddings.mean(dim=1)
        h = self.enc_to_dec_h(mean_img_embeddings)
        c = self.enc_to_dec_c(mean_img_embeddings)
        return h, c

    def decode_to_end(self, img_embedding, n_vocab, start_token_idx, end_token_idx, max_seq_len=50):
        last_token = start_token_idx
        caption = [last_token]

        img_embedding = img_embedding.view(1, self.enc_hidden_dim, -1).permute(0, 2, 1)
        h, c = self.init_hidden_cell_states(img_embedding)
        while len(caption) < max_seq_len and last_token != end_token_idx: 
            # Construct input
            word = torch.tensor([last_token], dtype=torch.long).to(self.device)
            caption_emb = self.emb(word)

            # Predict next character
            with torch.no_grad():
                att_img_emb = self.attention(img_embedding, h)
                inp = torch.cat([caption_emb, att_img_emb], dim=1) 
                h, c = self.lstm_cell(inp, (h, c))
                pred = self.fc(h)
                prob = F.softmax(pred, dim=1)[0]

            # Pick word based on probability instead of always picking the highest value
            word_idx = np.random.choice(list(range(n_vocab)), p=prob.numpy())

            caption.append(word_idx)
            last_token = word_idx
        return caption

    def forward(self, img_embeddings, captions, caption_lengths):
        batch_size = img_embeddings.size(0)

        # Flatten image, then permute dimensions
        img_embeddings = img_embeddings.view(batch_size, self.enc_hidden_dim, -1).permute(0, 2, 1)

        # Sort captions by decreasing lengths, and create embeddings
        caption_lengths, sort_idx = caption_lengths.sort(descending=True)
        img_embeddings = img_embeddings[sort_idx]
        captions = captions[sort_idx]
        caption_embeddings = self.emb(captions)

        # Minus 1 in length to avoid decoding at <end> token
        caption_lengths = (caption_lengths - 1).tolist()
        max_caption_length = max(caption_lengths)

        # Init hidden and cell states
        h, c = self.init_hidden_cell_states(img_embedding)

        # Init prediction as tensor
        predictions = torch.zeros(batch_size, max_caption_length, self.n_vocab).to(self.device)

        # Iterate and output each timestep of the captions
        for t in range(max_caption_length):
            # batch size at this timestep
            batch_size_t = sum([l > t for l in caption_lengths])

            # don't decode for captions who already ended
            caption_emb = caption_embeddings[:batch_size_t, t]
            img_emb = img_embeddings[:batch_size_t]
            h = h[:batch_size_t]
            c = c[:batch_size_t]

            # attention
            att_img_emb = self.attention(img_emb, h)

            # concatenate caption input and attention
            inp = torch.cat([caption_emb, att_img_emb], dim=1) 

            # decode
            h, c = self.lstm_cell(inp, (h, c))
            predictions[:batch_size_t, t] = self.fc(self.dropout(h))

        return predictions, captions, caption_lengths, sort_idx

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim):
        super(Attention, self).__init__()

        # Need encoder output to be in same dimension as decoder hidden state 
        self.enc_to_dec_dim = nn.Linear(encoder_dim, decoder_dim)

        # Layers for computing attention scores and weights
        self.att_scores = nn.Linear(decoder_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_h):
        # For each pixel in encoder output, map their embedding to the same dimension as decoder hidden state 
        encoder_emb = self.enc_to_dec_dim(encoder_out)

        # Compute attention scores by associating encoder embeddings for each pixel and decoder hidden state
        att_scores = self.att_scores(self.relu(encoder_emb + decoder_h.unsqueeze(1)))

        # Compute weights by doing softmax over attention scores of all pixels
        alpha = self.softmax(att_scores)

        # Weighted attention output
        attention_output = (encoder_out * alpha).sum(dim=1)

        return attention_output
