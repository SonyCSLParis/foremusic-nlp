# -*- coding: utf-8 -*-
""" 
Saving embeddings from trained models
"""
import os
import click
import math
import torch
from tqdm import tqdm
from joblib import load
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from src.helpers import get_embeddings
from src.models.ten_dim_regression import DimensionReductionRegressionHFModel

def read_csv(path: str) -> pd.DataFrame:
    """ Opening pandas, removing "Unnamed: 0" column """
    df = pd.read_csv(path)
    df = df[[col for col in df.columns if "Unnamed: 0" not in col]]
    return df

BASE_TOKENIZER = "xlm-roberta-base"
# base_model = "models/roberta-fine-tuned-regression/checkpoint-7200"
# data = "data/2023_09_28/train.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# save_path = "embeddings.npy"
# pca_model = "pca_model.joblib"
BATCH_SIZE = 16


@click.command()
@click.option("--base_model", help="Base model to get embeddings")
@click.option("--data", help="Path to .csv data to embed")
@click.option("--save", help="Save path for embeddings, in .npy format")
@click.option("--pca_model", help="PCA path model, a .joblib file")
def main(base_model, data, save, pca_model=None):
    tokenizer = AutoTokenizer.from_pretrained(BASE_TOKENIZER)
    
    if base_model.endswith(".pth"):
        model = torch.load(base_model)
    else:
        model = AutoModel.from_pretrained(base_model)
    df = read_csv(data)

    
    model.to(DEVICE)
    torch.cuda.empty_cache()
    embeddings = get_embeddings(model=model, tokenizer=tokenizer, sentences=df.pre_processed.values.tolist(), batch_size=BATCH_SIZE)

    if pca_model:
        pca = load(pca_model)
        embeddings = pca.transform(embeddings) 

    save_folder = "/".join(save.split("/")[:-1])
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    np.save(save, embeddings)


if __name__ == '__main__':
    main()
