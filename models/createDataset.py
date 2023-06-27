import torch
from transformers import BertTokenizer, BertModel, BertConfig



class HiddenStateDataset(torch.utils.data.Dataset):
    def __init__(self, df, bert_model, bert_tokenizer, device, bert_layer = 9, max_len = 30):
        self.df = df
        self.max_len = max_len
        self.tokenizer = bert_tokenizer
        self.bert = bert_model
        self.bert_layer = bert_layer
        self.device = device

    def __getitem__(self, index):
        text = self.df.iloc[index]["text"]
        label = self.df.iloc[index]["label"]
        
        encoded_input = self.tokenizer(
                str(text),
                padding='max_length', 
                truncation=True, 
                max_length= self.max_len,
                return_tensors="pt",
            )
        encoded_input = encoded_input.to(self.device)
                
        with torch.no_grad():
            model_outputs = self.bert(**encoded_input)
        
        outs = model_outputs[2][self.bert_layer][0,:,:].detach().cpu()
        return outs, torch.tensor(label)

    def __len__(self):
        return self.df.shape[0]
    


class SarcasticDataset(torch.utils.data.Dataset):
    def __init__(self, df, max_len=30):
        self.df = df
        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __getitem__(self, index):
        text = self.df.iloc[index]["text"]
        label = int(self.df.iloc[index]["label"])
        encoded_input = self.tokenizer.encode_plus(
                str(text),
                padding='max_length', 
                truncation=True, 
                max_length= self.max_len,
            )
        
        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"] if "attention_mask" in encoded_input else None
        token_type_ids = encoded_input["token_type_ids"] if "token_type_ids" in encoded_input else None

        data_input = {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "token_type_ids": torch.tensor(token_type_ids),
            "label": torch.tensor(label),
        }
        return data_input["input_ids"], data_input["attention_mask"], data_input["token_type_ids"], data_input["label"]

    def __len__(self):
        return self.df.shape[0]



class AugSarcasticDataset(torch.utils.data.Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __getitem__(self, index):
        tgt_input_ids, tgt_attention_mask, tgt_token_type_ids, aug_input_ids, \
        aug_attention_mask, aug_token_type_ids, label = self.ds[index]
  
        return tgt_input_ids, tgt_attention_mask, tgt_token_type_ids, aug_input_ids, \
        aug_attention_mask, aug_token_type_ids, label

    def __len__(self):
        return len(self.ds)
    