# -*- coding: utf-8 -*-
""" 
Saving embeddings from trained models
"""
import math
import torch
from tqdm import tqdm
from joblib import load
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from src.helpers import get_embeddings

def read_csv(path: str) -> pd.DataFrame:
    """ Opening pandas, removing "Unnamed: 0" column """
    df = pd.read_csv(path)
    df = df[[col for col in df.columns if "Unnamed: 0" not in col]]
    return df

base_tokenizer = "xlm-roberta-base"
base_model = "models/roberta-fine-tuned-regression/checkpoint-7200"
data = "data/2023_09_28/train.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_path = "embeddings.npy"
pca_model = "pca_model.joblib"
batch_size = 16



def main():
    tokenizer = AutoTokenizer.from_pretrained(base_tokenizer)
    model = AutoModel.from_pretrained(base_model)
    df = read_csv(data)

    
    model.to(device)
    torch.cuda.empty_cache()
    embeddings = get_embeddings(model=model, tokenizer=tokenizer, sentences=df.pre_processed.values.tolist(), batch_size=batch_size)

    if pca_model:
        pca = load(pca_model)
        embeddings = pca.transform(embeddings) 
    np.save(save_path, embeddings)


if __name__ == '__main__':
    main()
