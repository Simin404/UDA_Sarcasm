import json
import csv
import numpy as np
import pandas as pd
import torch
from createDataset import SarcasticDataset, HiddenStateDataset
from transformers import BertModel, BertConfig, BertTokenizer
from torch.utils.data import random_split

def remove_tabs(x):
    return x.replace("\t", " ").replace("\n", ". ")

def load_json(jfile):
    json_df = pd.read_json(jfile, lines=True)
    # remove irrelevant reviews
    filterd_json_df = json_df[json_df['headline'].apply(lambda x: isinstance(x, str))]
    filterd_json_df["headline"] = filterd_json_df["headline"].apply(remove_tabs)
    filterd_json_df = filterd_json_df[['headline', 'is_sarcastic']]
    filterd_json_df.rename(columns = {'headline':'text'}, inplace = True)
    filterd_json_df.rename(columns = {'is_sarcastic':'label'}, inplace = True)
    return filterd_json_df.dropna()

def load_txt(tfile):
    with open(tfile, 'r') as fd:
      data = [l.strip().split('\t') for l in fd.readlines()][1:]
    y = np.array([int(d[1]) for d in data])
    X = [d[2:] for d in data]
    X = np.array([''.join(str(e) for e in d) for d in X])
    new = np.transpose(np.asarray([X,y]))
    return pd.DataFrame(new, columns=['text', 'label'])

def load_csv(cfile):
  data = []
  with open(cfile, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        text, sarcastic = row['text'], row['label']
        data.append([text, sarcastic])
    return pd.DataFrame(data, columns=['text','label'])


def dataset_to_tsv(file_path, num = 6):
    """ Function that loads all five datasets
    """
    if num >= 1: 
        ghosh1 = load_csv(file_path + 'ghosh/train.csv')
        # headline1.to_csv("../data/headline/headline1_train.tsv", sep= "\t", index = False)
        ghosh2 = load_csv(file_path + 'ghosh/dev.csv')
        ghosh3 = load_csv(file_path + 'ghosh/test.csv')
        all = [ghosh1, ghosh2, ghosh3]
        ghosh = pd.concat(all)
        ghosh.to_csv(file_path + "ghosh/ghosh.tsv", sep= "\t", index = False)
        print("LOADING from database GHOSH: DONE")
        if num == 1:
            return ghosh
    else:
        print("Invalid number of datasets, cannot loading")
    if num >= 2: 
        semeval = load_txt(file_path + 'semeval/SemEval2018-T3-train-taskA.txt')
        semeval.to_csv(file_path + "semeval/semeval.tsv", sep= "\t", index = False)
        print("LOADING from database SemEval: DONE")
        if num == 2:
            return ghosh, semeval
    if num >=3 : 
        political = load_csv(file_path + 'political/PoliticalDebate-GEN.csv')
        political.to_csv(file_path + "political/political.tsv", sep= "\t", index = False)
        print("LOADING from database IAC(political debate): DONE")
        if num == 3:
            return ghosh, semeval, political
    if num >=4:
        ptacek = load_csv(file_path + 'ptacek/Ptacek_train_balanced.csv')
        ptacek.to_csv(file_path + "ptacek/ptacek.tsv", sep= "\t", index = False)
        print("LOADING from database Twitter(Ptacek): DONE")
        if num == 4:
            return ghosh, semeval, political, ptacek
    if num >= 5:
        sarc = load_csv(file_path + 'sarc/SARC-balance.csv')
        sarc['text'] = sarc['text'].astype('str')
        # only keep data len from [27, 30], which is 53756 data and keep 25000 for each label(0/1)
        mask2 = (sarc['text'].str.len() >= 27) & (sarc['text'].str.len() <= 30)
        df = sarc.loc[mask2]
        sarc['label'] = sarc['label'].astype('int32')
        df.to_csv(file_path + "sarc/sarc.tsv", sep= "\t", index = False)
        print("LOADING from database SARC: DONE")
        if num == 5:
            return ghosh, semeval, political, ptacek, sarc
    if num >= 6:   
        isarcasm1 = load_csv(file_path + 'isarcasm/train.csv')
        isarcasm2 = load_csv(file_path + 'isarcasm/dev.csv')
        isarcasm3 = load_csv(file_path + 'isarcasm/test.csv')
        all = [isarcasm1, isarcasm2, isarcasm3]
        isarcasm = pd.concat(all)
        isarcasm.to_csv(file_path + "isarcasm/isarcasm.tsv", sep= "\t", index = False)
        print("LOADING from database iSarcasm: DONE")
        if num == 6:
            return ghosh, semeval, political, ptacek, sarc, isarcasm
    else:
        print("Invalid number of datasets, cannot loading")


def load_all_datasets(file_path):
    """ Function that loads all six datasets
    """
    ghosh = pd.read_csv(file_path + "ghosh/ghosh.tsv", sep= "\t")
    semeval = pd.read_csv(file_path + "semeval/semeval.tsv", sep= "\t")
    political = pd.read_csv(file_path + 'political/political.tsv', sep= "\t")
    ptacek = pd.read_csv(file_path + "ptacek/ptacek.tsv", sep= "\t")
    sarc = pd.read_csv(file_path + "sarc/sarc.tsv", sep= "\t")
    isarcasm = pd.read_csv(file_path + "iSarcasm/isarcasm.tsv", sep= "\t")
    return ghosh, semeval, political, ptacek, sarc, isarcasm

def load_aug_datasets(file_path):
    """ Function that loads all three augment datasets
    """
    aug_semeval = pd.read_csv(file_path + "semeval/aug_semeval.tsv", sep= "\t")
    aug_political = pd.read_csv(file_path + 'political/aug_political.tsv', sep= "\t")
    aug_isarcasm = pd.read_csv(file_path + "iSarcasm/aug_isarcasm.tsv", sep= "\t")
    return aug_semeval, aug_political, aug_isarcasm

def load_single_dataset(dataset_name, data_path):
    dataset_name = dataset_name.lower()
    if dataset_name == 'ghosh': 
        data = pd.read_csv(data_path + "ghosh/ghosh.tsv", sep= "\t")
        print("LOADING database ghosh: Size:{}".format(data.shape))
        return data
    elif dataset_name == 'semeval': 
        data = pd.read_csv(data_path + "semeval/semeval.tsv", sep= "\t")
        print("LOADING database SemEval: Size:{}".format(data.shape))
        return data
    elif dataset_name == 'political': 
        data = pd.read_csv(data_path + 'political/political.tsv', sep= "\t")
        print("LOADING database IAC(political debate): Size:{}".format(data.shape))
        return data
    elif dataset_name == 'ptacek': 
        data = pd.read_csv(data_path + "ptacek/ptacek.tsv", sep= "\t")
        print("LOADING database Twitter(Ptacek): Size:{}".format(data.shape))
        return data
    elif dataset_name == 'sarc': 
        data = pd.read_csv(data_path + "sarc/sarc.tsv", sep= "\t")
        print("LOADING database SARC: Size:{}".format(data.shape))
        return data
    elif dataset_name == 'isarcasm': 
        data = pd.read_csv(data_path + "iSarcasm/isarcasm.tsv", sep= "\t")
        print("LOADING database iSarcasm: Size:{}".format(data.shape))
        return data
    else:
        print("Invalid dataset name, Only support ghosh, semeval, political, ptacek, sarc, isarcasm")


def print_info(ghosh, ptacek, sarc, semeval, political,isarcasm):
    s1 = ghosh.groupby('label').count()
    s1 = s1.rename(columns={"text":"ghosh"}, inplace = True)
    s2 = ptacek.groupby('label').count()
    s2 = s2.rename(columns={"text":"ptacek"}, inplace = True)
    s3 = sarc.groupby('label').count()
    s3 = s3.rename(columns={"text":"sarc"}, inplace = True)

    s4 = semeval.groupby('label').count()
    s4 = s4.rename(columns={"text":"semeval"}, inplace = True)
    s5 = political.groupby('label').count()
    s5 = s5.rename(columns={"text":"political"}, inplace = True)
    s6 = isarcasm.groupby('label').count()
    s6 = s6.rename(columns={"text":"isarcasm"}, inplace = True)
    all_info = pd.concat([s1, s2, s3, s4, s5, s6], axis=1)
    return all_info

def print_info(ghosh, semeval, political, ptacek, sarc, isarcasm):
    s1 = ghosh.groupby('label').count()
    s1 = s1.rename(columns={"text":"ghosh"})
    s2 = semeval.groupby('label').count()
    s2 = s2.rename(columns={"text":"semeval"})
    s3 = political.groupby('label').count()
    s3 = s3.rename(columns={"text":"political"})
    s4 = ptacek.groupby('label').count()
    s4 = s4.rename(columns={"text":"ptacek"})
    s5 = sarc.groupby('label').count()
    s5 = s5.rename(columns={"text":"sarc"})
    s6 = isarcasm.groupby('label').count()
    s6 = s6.rename(columns={"text":"isarcasm"})
    all_info = pd.concat([s1, s2, s3, s4, s5, s6], axis=1)
    return all_info

def multi_dataset(data_list, name_list, ratio = 0.8):
    data_dic = {}
    if ratio != 1.0:
        for i in range(len(name_list)):
            load_dataset = SarcasticDataset(data_list[i])
            train, val = random_split(load_dataset, [ratio, 1 - ratio]) 
            data_dic[name_list[i]+'_train'] = train
            data_dic[name_list[i]+'_val'] = val 
    else:
        for i in range(len(name_list)):
            load_dataset = SarcasticDataset(data_list[i])
            data_dic[name_list[i]+'_train'] = load_dataset
    return data_dic

def aug_multi_dataset(data_list, aug_list, name_list, ratio = 0.8):
    data_dic = {}
    if ratio != 1.0:
        for i in range(len(name_list)):
            load_dataset = SarcasticDataset(data_list[i])
            train, val = random_split(load_dataset, [ratio, 1 - ratio]) 
            data_dic[name_list[i]+'_train'] = train
            data_dic[name_list[i]+'_val'] = val 
    else:
        for i in range(len(name_list)):
            load_dataset = SarcasticDataset(data_list[i])
            aug_dataset = SarcasticDataset(aug_list[i])
            data_dic[name_list[i]+'_train'] = aug_dataset
            data_dic[name_list[i]+'_val'] = load_dataset
    return data_dic

def multi_Hidden_dataset(data_list, name_list, device,  layer = 9, max_len = 30, ratio = 0.8):
    data_dic = {}
    tokenizer_data = BertTokenizer.from_pretrained('bert-base-uncased')
    config_data = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
    model_data = BertModel.from_pretrained('bert-base-uncased', config = config_data)
    model_data = model_data.to(device)
    if ratio != 1.0:
        for i in range(len(name_list)):
            load_dataset = HiddenStateDataset(data_list[i], model_data, tokenizer_data, device, layer, max_len = 30)
            train, val = random_split(load_dataset, [ratio, 1 - ratio]) 
            data_dic[name_list[i]+'_train'] = train
            data_dic[name_list[i]+'_val'] = val 
    else:
        for i in range(len(name_list)):
            load_dataset = HiddenStateDataset(data_list[i], model_data, tokenizer_data, device, layer, max_len = 30)
            data_dic[name_list[i]+'_train'] = load_dataset
    return data_dic

def aug_multi_Hidden_dataset(data_list, aug_list, name_list, device, layer = 9, max_len = 30, ratio = 0.8):
    data_dic = {}
    tokenizer_data = BertTokenizer.from_pretrained('bert-base-uncased')
    config_data = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
    model_data = BertModel.from_pretrained('bert-base-uncased', config = config_data)
    model_data = model_data.to(device)
    if ratio != 1.0:
        for i in range(len(name_list)):
            load_dataset = HiddenStateDataset(data_list[i], model_data, tokenizer_data, device, layer, max_len = 30)
            train, val = random_split(load_dataset, [ratio, 1 - ratio]) 
            data_dic[name_list[i]+'_train'] = train
            data_dic[name_list[i]+'_val'] = val 
    else:
        for i in range(len(name_list)):
            load_dataset = HiddenStateDataset(data_list[i], model_data, tokenizer_data, device, layer, max_len = 30)
            aug_dataset = HiddenStateDataset(aug_list[i], model_data, tokenizer_data, device, layer, max_len = 30)
            data_dic[name_list[i]+'_train'] = aug_dataset
            data_dic[name_list[i]+'_val'] = load_dataset
    return data_dic

def aug_multi_Sarcastic_dataset(data_list, aug_list, name_list, ratio = 0.8):
    data_dic = {}
    if ratio != 1.0:
        for i in range(len(name_list)):
            load_dataset = SarcasticDataset(data_list[i], max_len = 30)
            train, val = random_split(load_dataset, [ratio, 1 - ratio]) 
            data_dic[name_list[i]+'_train'] = train
            data_dic[name_list[i]+'_val'] = val 
    else:
        for i in range(len(name_list)):
            load_dataset = SarcasticDataset(data_list[i], max_len = 30)
            aug_dataset = SarcasticDataset(aug_list[i], max_len = 30)
            data_dic[name_list[i]+'_train'] = aug_dataset
            data_dic[name_list[i]+'_val'] = load_dataset
    return data_dic