# -*- coding: utf-8 -*-
""" 
Generic helpers
"""
import torch
import math
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_function(tokenizer, examples, target):
    """ Pre-processing text and score """
    label = examples[target] 
    examples = tokenizer(examples["pre_processed"], truncation=True, padding="max_length", max_length=256)

    # Change this to real number -> removed because already float
    examples["label"] = float(label)
    return examples

def get_data(tokenizer, train_ds, eval_ds, target):
    """ Format data to be re-usable for training """
    ds = {"train": train_ds, "eval": eval_ds}
    remove_columns = ['Unnamed: 0', 'track_id', 'start_date', 'release_date', 'artist_id', 'artist_name', 'album_id', 'album_name', 'ytv_ref', 'ytc_id', 'db_country', 'lyrics', 'train_eval_test']
    for split in ds:
        ds[split] = ds[split].map(lambda x: preprocess_function(tokenizer, x, target), remove_columns=remove_columns)
    return ds

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def check_config(keys, config):
    """ Check that all parameters are here """
    if any(x not in config for x in keys):
        raise ValueError(f"All the following elements must be in config: {keys}")

def get_embeddings(model, tokenizer, sentences, batch_size: int = 16):
    """ Retrieve embeddings in batches """
    res = []
    nb_batch = math.ceil(len(sentences)/batch_size)
    for i in tqdm(range(0, nb_batch)):
        batch_sentences = sentences[i*batch_size:(i+1)*batch_size]
        batch_sentences = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
        # print(batch_sentences["input_ids"].shape, batch_sentences["attention_mask"].shape)
        batch_sentences = batch_sentences.to(DEVICE)
        if "DimensionReductionRegressionHFModel" in str(type(model)):
            input_ids = batch_sentences["input_ids"].to(DEVICE)
            attention_mask = torch.ones_like(input_ids)
            attention_mask[input_ids == tokenizer.pad_token_id] = 0
            inputs = {"input_ids": input_ids, "attention_mask": attention_mask.to(DEVICE)}
            with torch.no_grad():
                outputs = model(**inputs)
            embeddings = outputs['last_hidden_state']
        else:
            inputs = batch_sentences
            with torch.no_grad():
                outputs = model(**inputs)
            embeddings = mean_pooling(outputs, inputs['attention_mask'])
        with torch.no_grad():
            outputs = model(**inputs)
            
        embeddings = F.normalize(embeddings, p=2, dim=1)
        res.append(embeddings.cpu().numpy())

    # Concatenate the batches
    return np.concatenate(res, axis=0)

def get_embeddings_old(model, tokenizer, sentences, batch_size: int = 16):
    """ Retrieve embeddings in batches """
    res = []
    nb_batch = math.ceil(len(sentences)/batch_size)
    for i in tqdm(range(0, nb_batch)):
        batch_sentences = sentences[i*batch_size:(i+1)*batch_size]
        batch_sentences = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
        if "DimensionReductionRegressionHFModel" in str(type(model)):
            input_ids = batch_sentences.to(DEVICE)
        else:
            input_ids = batch_sentences["input_ids"].to(DEVICE)
        attention_mask = torch.ones_like(input_ids)
        attention_mask[input_ids == tokenizer.pad_token_id] = 0
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask.to(DEVICE)}
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs['last_hidden_state']
        if len(embeddings.shape) == 2:
            embeddings = embeddings.cpu().numpy()
        else:
            embeddings = torch.mean(embeddings, dim=1).cpu().numpy()
        # embeddings = F.normalize(torch.from_numpy(embeddings), p=2, dim=-1).numpy()
        res.append(embeddings)

    # Concatenate the batches
    return np.concatenate(res, axis=0)

def read_csv(path: str) -> pd.DataFrame:
    """ Opening pandas, removing "Unnamed: 0" column """
    df = pd.read_csv(path)
    df = df[[col for col in df.columns if "Unnamed: 0" not in col]]
    return df