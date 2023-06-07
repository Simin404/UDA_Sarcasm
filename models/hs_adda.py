import torch.nn.functional as F
from torch import nn
from transformers import BertModel
import copy
from gradient import GradientReversalFn

class HS_ADDN_Discriminator_0(nn.Module):
    def __init__(self):
        super(HS_ADDN_Discriminator_0, self).__init__()
        bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert_pooler = copy.deepcopy(bert.pooler)
        self.hidden2 = nn.Linear(768, 768*2)
        self.hidden3 = nn.Linear(768*2, 768)
        self.dropout = nn.Dropout(0.1)
        self.output = nn.Linear(768, 2)
        self.act = nn.Tanh()

    def forward(self, x , grl_lambda = 1.0):
        x = self.bert_pooler(x)
        x = self.act(self.hidden2(x))
        x = self.dropout(x)
        x = self.act(self.hidden3(x))
        x = self.dropout(x)
        return self.output(x)


class HS_ADDN_Classifier_0(nn.Module):
    def __init__(self):
        super(HS_ADDN_Classifier_0, self).__init__()
        bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert_encoder2 = copy.deepcopy(bert.encoder.layer[1])
        self.bert_encoder3 = copy.deepcopy(bert.encoder.layer[2])
        self.bert_encoder4 = copy.deepcopy(bert.encoder.layer[3])
        self.bert_encoder5 = copy.deepcopy(bert.encoder.layer[4])
        self.bert_encoder6 = copy.deepcopy(bert.encoder.layer[5])
        self.bert_encoder7 = copy.deepcopy(bert.encoder.layer[6])
        self.bert_encoder8 = copy.deepcopy(bert.encoder.layer[7])
        self.bert_encoder9 = copy.deepcopy(bert.encoder.layer[8])
        self.bert_encoder10 = copy.deepcopy(bert.encoder.layer[9])
        self.bert_encoder11 = copy.deepcopy(bert.encoder.layer[10])
        self.bert_encoder12 = copy.deepcopy(bert.encoder.layer[11])
        self.bert_pooler = copy.deepcopy(bert.pooler)
        self.hidden1 = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.1)
        self.output = nn.Linear(768, 2)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.bert_encoder2(x)[0]
        x = self.bert_encoder3(x)[0]
        x = self.bert_encoder4(x)[0]
        x = self.bert_encoder5(x)[0]
        x = self.bert_encoder6(x)[0]
        x = self.bert_encoder7(x)[0]
        x = self.bert_encoder8(x)[0]
        x = self.bert_encoder9(x)[0]
        x = self.bert_encoder10(x)[0]
        x = self.bert_encoder11(x)[0]
        x = self.bert_encoder12(x)[0]
        x = self.bert_pooler(x)
        x = self.dropout(x)
        x = self.act(self.hidden1(x))
        return self.output(x)


class HS_ADDN_Encoder_0(nn.Module):
    def __init__(self):
        super(HS_ADDN_Encoder_0, self).__init__()
        bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert_encoder2 = copy.deepcopy(bert.encoder.layer[1])
        self.bert_encoder3 = copy.deepcopy(bert.encoder.layer[2])

    def forward(self, x):
        x = self.bert_encoder2(x)[0]
        x = self.bert_encoder3(x)[0]
        return x



class HS_ADDN_Classifier_1(nn.Module):
    def __init__(self):
        super(HS_ADDN_Classifier_1, self).__init__()
        bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert_encoder4 = copy.deepcopy(bert.encoder.layer[3])
        self.bert_encoder5 = copy.deepcopy(bert.encoder.layer[4])
        self.bert_encoder6 = copy.deepcopy(bert.encoder.layer[5])
        self.bert_encoder7 = copy.deepcopy(bert.encoder.layer[6])
        self.bert_encoder8 = copy.deepcopy(bert.encoder.layer[7])
        self.bert_encoder9 = copy.deepcopy(bert.encoder.layer[8])
        self.bert_encoder10 = copy.deepcopy(bert.encoder.layer[9])
        self.bert_encoder11 = copy.deepcopy(bert.encoder.layer[10])
        self.bert_encoder12 = copy.deepcopy(bert.encoder.layer[11])
        self.bert_pooler = copy.deepcopy(bert.pooler)
        self.hidden1 = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.1)
        self.output = nn.Linear(768, 2)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.bert_encoder4(x)[0]
        x = self.bert_encoder5(x)[0]
        x = self.bert_encoder6(x)[0]
        x = self.bert_encoder7(x)[0]
        x = self.bert_encoder8(x)[0]
        x = self.bert_encoder9(x)[0]
        x = self.bert_encoder10(x)[0]
        x = self.bert_encoder11(x)[0]
        x = self.bert_encoder12(x)[0]
        x = self.bert_pooler(x)
        x = self.act(self.hidden1(x))
        x = self.dropout(x)
        return self.output(x)


class HS_ADDN_Encoder_1(nn.Module):
    def __init__(self):
        super(HS_ADDN_Encoder_1, self).__init__()
        bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert_encoder2 = copy.deepcopy(bert.encoder.layer[1])
        self.bert_encoder3 = copy.deepcopy(bert.encoder.layer[2])

    def forward(self, x):
        x = self.bert_encoder2(x)[0]
        x = self.bert_encoder3(x)[0]
        return x

    

class HS_ADDN_Classifier_3(nn.Module):
    def __init__(self):
        super(HS_ADDN_Classifier_3, self).__init__()
        bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert_encoder6 = copy.deepcopy(bert.encoder.layer[5])
        self.bert_encoder7 = copy.deepcopy(bert.encoder.layer[6])
        self.bert_encoder8 = copy.deepcopy(bert.encoder.layer[7])
        self.bert_encoder9 = copy.deepcopy(bert.encoder.layer[8])
        self.bert_encoder10 = copy.deepcopy(bert.encoder.layer[9])
        self.bert_encoder11 = copy.deepcopy(bert.encoder.layer[10])
        self.bert_encoder12 = copy.deepcopy(bert.encoder.layer[11])
        self.bert_pooler = copy.deepcopy(bert.pooler)
        self.hidden1 = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.1)
        self.output = nn.Linear(768, 2)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.bert_encoder6(x)[0]
        x = self.bert_encoder7(x)[0]
        x = self.bert_encoder8(x)[0]
        x = self.bert_encoder9(x)[0]
        x = self.bert_encoder10(x)[0]
        x = self.bert_encoder11(x)[0]
        x = self.bert_encoder12(x)[0]
        x = self.bert_pooler(x)
        x = self.act(self.hidden1(x))
        x = self.dropout(x)
        return self.output(x)


class HS_ADDN_Encoder_3(nn.Module):
    def __init__(self):
        super(HS_ADDN_Encoder_3, self).__init__()
        bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert_encoder4 = copy.deepcopy(bert.encoder.layer[3])
        self.bert_encoder5 = copy.deepcopy(bert.encoder.layer[4])

    def forward(self, x):
        x = self.bert_encoder4(x)[0]
        x = self.bert_encoder5(x)[0]
        return x




class HS_ADDN_Classifier_5(nn.Module):
    def __init__(self):
        super(HS_ADDN_Classifier_5, self).__init__()
        bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert_encoder8 = copy.deepcopy(bert.encoder.layer[7])
        self.bert_encoder9 = copy.deepcopy(bert.encoder.layer[8])
        self.bert_encoder10 = copy.deepcopy(bert.encoder.layer[9])
        self.bert_encoder11 = copy.deepcopy(bert.encoder.layer[10])
        self.bert_encoder12 = copy.deepcopy(bert.encoder.layer[11])
        self.bert_pooler = copy.deepcopy(bert.pooler)
        self.hidden1 = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.1)
        self.output = nn.Linear(768, 2)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.bert_encoder8(x)[0]
        x = self.bert_encoder9(x)[0]
        x = self.bert_encoder10(x)[0]
        x = self.bert_encoder11(x)[0]
        x = self.bert_encoder12(x)[0]
        x = self.bert_pooler(x)
        x = self.act(self.hidden1(x))
        x = self.dropout(x)
        return self.output(x)


class HS_ADDN_Encoder_5(nn.Module):
    def __init__(self):
        super(HS_ADDN_Encoder_5, self).__init__()
        bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert_encoder6 = copy.deepcopy(bert.encoder.layer[5])
        self.bert_encoder7 = copy.deepcopy(bert.encoder.layer[6])

    def forward(self, x):
        x = self.bert_encoder6(x)[0]
        x = self.bert_encoder7(x)[0]
        return x

    


class HS_ADDN_Classifier_7(nn.Module):
    def __init__(self):
        super(HS_ADDN_Classifier_7, self).__init__()
        bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert_encoder10 = copy.deepcopy(bert.encoder.layer[9])
        self.bert_encoder11 = copy.deepcopy(bert.encoder.layer[10])
        self.bert_encoder12 = copy.deepcopy(bert.encoder.layer[11])
        self.bert_pooler = copy.deepcopy(bert.pooler)
        self.hidden1 = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.1)
        self.output = nn.Linear(768, 2)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.bert_encoder10(x)[0]
        x = self.bert_encoder11(x)[0]
        x = self.bert_encoder12(x)[0]
        x = self.bert_pooler(x)
        x = self.act(self.hidden1(x))
        x = self.dropout(x)
        return self.output(x)


class HS_ADDN_Encoder_7(nn.Module):
    def __init__(self):
        super(HS_ADDN_Encoder_7, self).__init__()
        bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert_encoder8 = copy.deepcopy(bert.encoder.layer[7])
        self.bert_encoder9 = copy.deepcopy(bert.encoder.layer[8])

    def forward(self, x):
        x = self.bert_encoder8(x)[0]
        x = self.bert_encoder9(x)[0]
        return x

    
