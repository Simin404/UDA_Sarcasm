import torch.nn.functional as F
from torch import nn
from transformers import BertModel

class ADDN_Encoder(nn.Module):
    def __init__(self, cfg):
        super(ADDN_Encoder, self).__init__()
        self.cfg = cfg
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.encoder = nn.Sequential(
            nn.Linear(768, 768),
            nn.Tanh(),
        )

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # out = outputs[0][:,0,:] # cls
        out = outputs[1]  # pooler
        return out



class ADDN_Classifier(nn.Module):

    def __init__(self):
        super(ADDN_Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(768, 2)
        )  

    def forward(self, feat):
        out = self.classifier(feat)
        return out
    
class ADDN_Discriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self):
        """Init discriminator."""
        super(ADDN_Discriminator, self).__init__()

        self.restored = False

        self.layer = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Linear(384, 2),
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out