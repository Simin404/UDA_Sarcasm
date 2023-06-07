import torch
import torch.nn as nn
import torch.optim as optim
from gradient import GradientReversalFn
from transformers import BertModel

class DANN(nn.Module):
    def __init__(self, cfg):
        super(DANN, self).__init__()
        self.cfg = cfg
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(self.cfg["hidden_dropout_prob"])
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Linear(192, 96),
            nn.ReLU(),
            nn.Linear(96, 2),
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Linear(384, 96),
            nn.ReLU(),
            nn.Linear(96, 2),
        )


    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, grl_lambda = 1.0):

        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        reversed_pooled_output = GradientReversalFn.apply(pooled_output, grl_lambda)

        sentiment_pred = self.sentiment_classifier(pooled_output)
        domain_pred = self.domain_classifier(reversed_pooled_output)

        return sentiment_pred, domain_pred
    

