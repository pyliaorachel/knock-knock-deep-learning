import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class BertSentiment(nn.Module):
    def __init__(self, n_class, hidden_dim=256, dropout=0.2):
        super(BertSentiment, self).__init__()

        self.enc = BertModel.from_pretrained('bert-base-uncased')
        self.out = nn.Linear(self.enc.config.hidden_size, n_class)

    def freeze_bert(self, freeze):
        for param in self.enc.parameters():
            param.requires_grad = not freeze

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.enc(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        y = self.out(output.pooler_output)
        return y