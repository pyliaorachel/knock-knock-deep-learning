import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


class ImageEncoder(nn.Module):
    def __init__(self, device, pretrained=False):
        super(ImageEncoder, self).__init__()

        self.device = device

        # ResNet includes 2 extra layers for classification, but they're not needed since we only want the embedding
        resnet = models.resnet50(pretrained=pretrained)
        modules = list(resnet.children())[:-2]
        self.net = nn.Sequential(*modules)

        self.hidden_dim = 2048

    def freeze_pretrained(self, freeze):
        for param in self.net.parameters():
            param.requires_grad = not freeze

    def forward(self, imgs):
        y = self.net(imgs)
        return y

class CaptionDecoder(nn.Module):
    def __init__(self, device, n_vocab, embedding_dim=256, enc_hidden_dim=2048, dec_hidden_dim=256, att_dim=256, dropout=0.2, use_pretrained_emb=False, word_to_int=None):
        super(CaptionDecoder, self).__init__()

        self.device = device
        self.n_vocab = n_vocab
        self.embedding_dim = embedding_dim
        self.enc_hidden_dim = enc_hidden_dim
        self.dec_hidden_dim = dec_hidden_dim

        self.enc_to_dec_h = nn.Linear(enc_hidden_dim, dec_hidden_dim)
        self.enc_to_dec_c = nn.Linear(enc_hidden_dim, dec_hidden_dim)

        self.emb = self.init_emb(use_pretrained_emb, word_to_int)
        self.lstm_cell = nn.LSTMCell(embedding_dim + enc_hidden_dim, dec_hidden_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(dec_hidden_dim, n_vocab)

        self.attention = Attention(enc_hidden_dim, dec_hidden_dim, attention_dim=att_dim)

    def freeze_pretrained(self, freeze):
        for param in self.emb.parameters():
            param.requires_grad = not freeze

    def init_emb(self, use_pretrained_emb, word_to_int):
        emb = nn.Embedding(self.n_vocab, self.embedding_dim)
        if use_pretrained_emb:
            print('Loading pretrained word embeddings...')

            # Download pre-trained GloVe embeddings, turn into Word2Vec format
            glove_file = './data/glove.6B.{}d.txt'.format(self.embedding_dim)
            word2vec_glove_file = './data/glove.6B.{}d.word2vec.txt'.format(self.embedding_dim)
            if not os.path.isfile(word2vec_glove_file):
                glove2word2vec(glove_file, word2vec_glove_file)

            # Load model
            model = KeyedVectors.load_word2vec_format(word2vec_glove_file)

            # Construct pretrained embeddings according to our vocab
            # Clone weights so if some word not in glove, they are randomly initialized
            pretrained_emb = emb.weight.clone().detach()
            for word, i in word_to_int.items():
                if word in model:
                    pretrained_emb[i] = torch.tensor(model[word])

            # Load pretrained embeddings
            emb.load_state_dict({'weight': pretrained_emb})

            print('Done.')
        return emb

    def init_hidden_cell_states(self, img_embeddings):
        # Init hidden and cell states by taking mean of img embedding over all pixels
        mean_img_embeddings = img_embeddings.mean(dim=1)
        h = self.enc_to_dec_h(mean_img_embeddings)
        c = self.enc_to_dec_c(mean_img_embeddings)
        return h, c

    def decode_to_end(self, img_embedding, n_vocab, start_token_idx, end_token_idx, max_seq_len=40, k=1):
        # Beam search
        top_k_captions = [[start_token_idx]] * k
        top_k_scores = torch.zeros((k, 1))
        img_embedding = img_embedding.view(1, self.enc_hidden_dim, -1).permute(0, 2, 1)
        img_embeddings = img_embedding.expand(k, -1, -1)
        h, c = self.init_hidden_cell_states(img_embeddings)

        seq_t = 1
        completed_captions = []
        completed_caption_scores = []
        while seq_t < max_seq_len and len(top_k_scores) != 0:
            k = len(top_k_scores)

            # Construct input
            last_tokens = [s[-1] for s in top_k_captions]
            word = torch.tensor(last_tokens, dtype=torch.long)
            caption_emb = self.emb(word)

            # Predict
            with torch.no_grad():
                att_img_emb = self.attention(img_embeddings, h)
                inp = torch.cat([caption_emb, att_img_emb], dim=1) 
                h, c = self.lstm_cell(inp, (h, c))
                pred = self.fc(h)
                log_prob = F.log_softmax(pred, dim=1)

            # Expand the seq, pick k seq with best scores
            total_scores = top_k_scores.expand_as(log_prob) + log_prob
            top_k_scores, top_k_idx = total_scores.view(-1).topk(k)

            new_top_k_captions = []
            new_top_k_scores = []
            for seq_i, idx in enumerate(top_k_idx):
                idx = idx.item()
                prev = int(idx / n_vocab)
                nxt = idx % n_vocab
                expanded_seq = top_k_captions[prev] + [nxt]

                if nxt == end_token_idx:
                    completed_captions.append(expanded_seq)
                    completed_caption_scores.append(top_k_scores[seq_i] / (seq_t+1)) # normalize by seq length
                else:
                    new_top_k_captions.append(expanded_seq)
                    new_top_k_scores.append(top_k_scores[seq_i])
            top_k_captions = new_top_k_captions
            top_k_scores = torch.tensor(new_top_k_scores).unsqueeze(1)

            seq_t += 1

        # From all the completed seq, return the one with highest score as result
        # If no completed seq, return highest in remaining top k captions
        if len(completed_captions) != 0:
            max_idx = torch.argmax(torch.tensor(completed_caption_scores)).item()
            return completed_captions[max_idx.item()]
        else:
            max_idx = torch.argmax(top_k_scores.squeeze(1)).item()
            return top_k_captions[max_idx]

    def forward(self, img_embeddings, captions, caption_lengths, epsilon=1):
        batch_size = img_embeddings.size(0)

        # Flatten image, then permute dimensions
        img_embeddings = img_embeddings.view(batch_size, self.enc_hidden_dim, -1).permute(0, 2, 1)

        # Sort captions by decreasing lengths, and create embeddings
        caption_lengths, sort_idx = caption_lengths.sort(descending=True)
        img_embeddings = img_embeddings[sort_idx]
        captions = captions[sort_idx]
        caption_embeddings = self.emb(captions)

        # Init hidden and cell states
        h, c = self.init_hidden_cell_states(img_embeddings)

        # Minus 1 in length to avoid decoding at <end> token
        decode_lengths = (caption_lengths - 1).tolist()
        max_decode_length = max(decode_lengths)

        # Init prediction as tensor
        predictions = torch.zeros(batch_size, max_decode_length, self.n_vocab).to(self.device)

        # Iterate and output each timestep of the captions
        for t in range(max_decode_length):
            # Batch size at this timestep
            batch_size_t = sum([l > t for l in decode_lengths])

            # Don't decode for captions who already ended
            img_emb = img_embeddings[:batch_size_t]
            h = h[:batch_size_t]
            c = c[:batch_size_t]

            # Curriculum learning: prob of epsilon using real target as next timestep's input, (1-epsilon) choosing the best from previous output
            if t != 0 and np.random.random_sample() >= epsilon:
                last_tokens = torch.argmax(predictions[:batch_size_t, t-1], dim=1)
                caption_emb = self.emb(last_tokens)
            else:
                caption_emb = caption_embeddings[:batch_size_t, t]

            # Attention
            att_img_emb = self.attention(img_emb, h)

            # Concatenate caption input and attention
            inp = torch.cat([caption_emb, att_img_emb], dim=1) 

            # Decode
            h, c = self.lstm_cell(inp, (h, c))
            predictions[:batch_size_t, t] = self.fc(self.dropout(h))

        return predictions, captions, caption_lengths, sort_idx

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim=256):
        super(Attention, self).__init__()

        self.enc_to_att = nn.Linear(encoder_dim, attention_dim)
        self.dec_to_att = nn.Linear(decoder_dim, attention_dim)

        self.att_scores = nn.Linear(attention_dim, 1)

    def forward(self, encoder_out, decoder_h):
        # Map encoder output and decoder hidden state to attention dimension
        encoder_att = self.enc_to_att(encoder_out)
        decoder_att = self.dec_to_att(decoder_h)

        # Compute attention scores by associating encoder embeddings for each pixel and decoder hidden state
        att_scores = self.att_scores(F.relu(encoder_att + decoder_att.unsqueeze(1)))

        # Compute weights by doing softmax over attention scores of all pixels
        alpha = F.softmax(att_scores, dim=1)

        # Weighted attention output
        attention_output = (encoder_out * alpha).sum(dim=1)

        return attention_output
