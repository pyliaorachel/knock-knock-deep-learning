import csv

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class SentimentDataset(Dataset):
    def __init__(self, path, seq_length=512):
        super(SentimentDataset).__init__()
        self.path = path
        self.seq_length = seq_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.parse_corpus()

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return (self.input_ids[i], self.attention_mask[i], self.token_type_ids[i], self.y[i])

    def parse_corpus(self):
        '''
        Parse raw corpus text into input-output pairs, output tokenized by BertTokenizer
        '''
        input_ids, attention_mask, token_type_ids, y = [], [], [], []

        # Read tsv from file
        with open(self.path, 'r') as f:
            rd = csv.reader(f, delimiter='\t')
            next(rd) # ignore header
            for row in rd:
                phrase_id, sentence_id, phrase, sentiment = row

                # Tokenize each batch of phrases, truncate or pad to max length specified
                x = self.tokenizer(phrase, return_tensors='pt', max_length=self.seq_length, truncation=True, padding='max_length') # pt stands for PyTorch

                input_ids.append(x.input_ids.squeeze(0))
                attention_mask.append(x.attention_mask.squeeze(0))
                token_type_ids.append(x.token_type_ids.squeeze(0))
                y.append(int(sentiment))

        self.input_ids, self.attention_mask, self.token_type_ids, self.y = input_ids, attention_mask, token_type_ids, y
        self.n = len(y)