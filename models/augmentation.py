from transformers import MarianMTModel, MarianTokenizer
import torch
import tqdm
import pandas as pd


def aug_data(dataset, save_path, device, multi = 3):
    sentences=list(dataset.loc[:, "text"].values)
    n = 50
    sentences_lists = [sentences[i * n:(i + 1) * n] for i in range((len(sentences) + n - 1) // n )]
    fr_aug=[]
    de_aug=[]
    ru_aug=[]
    # Get english to German model
    en_de_model_name = 'Helsinki-NLP/opus-mt-en-de'
    en_de_tokenizer = MarianTokenizer.from_pretrained(en_de_model_name)
    en_de_model = MarianMTModel.from_pretrained(en_de_model_name).to(device)
    # Get German to english model
    de_en_model_name = 'Helsinki-NLP/opus-mt-de-en'
    de_en_tokenizer = MarianTokenizer.from_pretrained(de_en_model_name)
    de_en_model = MarianMTModel.from_pretrained(de_en_model_name).to(device)


    # Get english to russia model
    en_ru_model_name = 'Helsinki-NLP/opus-mt-en-ru'
    en_ru_tokenizer = MarianTokenizer.from_pretrained(en_ru_model_name)
    en_ru_model = MarianMTModel.from_pretrained(en_ru_model_name).to(device)
    # Get russia to english model
    ru_en_model_name = 'Helsinki-NLP/opus-mt-ru-en'
    ru_en_tokenizer = MarianTokenizer.from_pretrained(ru_en_model_name)
    ru_en_model = MarianMTModel.from_pretrained(ru_en_model_name).to(device)


    # Get english to france model
    en_fr_model_name = 'Helsinki-NLP/opus-mt-en-fr'
    en_fr_tokenizer = MarianTokenizer.from_pretrained(en_fr_model_name)
    en_fr_model = MarianMTModel.from_pretrained(en_fr_model_name).to(device)
    # Get france to english model
    fr_en_model_name = 'Helsinki-NLP/opus-mt-fr-en'
    fr_en_tokenizer = MarianTokenizer.from_pretrained(fr_en_model_name)
    fr_en_model = MarianMTModel.from_pretrained(fr_en_model_name).to(device)
    print(len(sentences_lists))
    i = 0
    for sen in sentences_lists:
        aug1=back_translation(device,sen, en_fr_model, en_fr_tokenizer, fr_en_model, fr_en_tokenizer)
        aug2=back_translation(device,sen, en_de_model, en_de_tokenizer, de_en_model, de_en_tokenizer)
        aug3=back_translation(device,sen, en_ru_model, en_ru_tokenizer, ru_en_model, ru_en_tokenizer)
        fr_aug.append(aug1)
        de_aug.append(aug2)
        ru_aug.append(aug3)
        i += 1
        print(i)
    
    fr =[item for sublist in fr_aug for item in sublist]
    de =[item for sublist in de_aug for item in sublist]
    ru =[item for sublist in ru_aug for item in sublist]

    aug1 = pd.DataFrame({'text':fr, 'label':dataset['label']})
    aug2 = pd.DataFrame({'text':de, 'label':dataset['label']})
    aug3 = pd.DataFrame({'text':ru, 'label':dataset['label']})
    aug_data = pd.concat([dataset, aug1, aug2, aug3], axis=0).reset_index(drop=True)
    aug_data.to_csv(save_path, sep="\t")
    print('Original shape:{}, After augmentation:{}'.format(dataset.shape, aug_data.shape))


def back_translation(device,texts, eng_model, eng_tokenizer, tmp_model, tmp_tokenizer, language="fr"):
    formated_batch_texts = [">>{}<< {}".format(language, text) for text in texts]
    new_language = eng_model.generate(**eng_tokenizer(formated_batch_texts, return_tensors="pt",max_length=50, padding=True).to(device))
    translated_texts = [eng_tokenizer.decode(t, skip_special_tokens=True) for t in new_language]
    back_eng = tmp_model.generate(**tmp_tokenizer(translated_texts, return_tensors="pt",max_length=50, padding=True).to(device))
    aug_texts = [tmp_tokenizer.decode(t, skip_special_tokens=True) for t in back_eng]
    return aug_texts