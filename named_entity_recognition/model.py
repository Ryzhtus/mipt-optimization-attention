import torch
from torch import nn
from transformers import BertModel

class BertNER(nn.Module):
    def __init__(self, n_classes):
        super(BertNER, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        encoded_layers, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = encoded_layers
        logits = self.fc(x)

        return logits
