# -*- coding: utf-8 -*-
""" 
Generic helpers
"""
import torch
import math
from tqdm import tqdm
import pandas as pd
import numpy as np

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

def check_config(keys, config):
    """ Check that all parameters are here """
    if any(x not in keysg for x in keys):
        raise ValueError(f"All the following elements must be in config: {keys}")

def get_embeddings(model, tokenizer, sentences, batch_size: int = 16):
    """ Retrieve embeddings in batches """
    res = []
    nb_batch = math.ceil(len(sentences)/batch_size)
    for i in tqdm(range(0, nb_batch)):
        batch_sentences = sentences[i*batch_size:(i+1)*batch_size]
        batch_sentences = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
        input_ids = batch_sentences["input_ids"].to(DEVICE)

        with torch.no_grad():
            outputs = model(input_ids)
        embeddings = outputs.last_hidden_state
        embeddings = torch.mean(embeddings, dim=1).cpu().numpy()
        res.append(embeddings)

    # Concatenate the batches
    return np.concatenate(res, axis=0)

def read_csv(path: str) -> pd.DataFrame:
    """ Opening pandas, removing "Unnamed: 0" column """
    df = pd.read_csv(path)
    df = df[[col for col in df.columns if "Unnamed: 0" not in col]]
    return df