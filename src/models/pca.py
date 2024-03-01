# -*- coding: utf-8 -*-
""" 
Fine-tuned model for regression, on the n-shaped embeddings
"""
import os
import json
import math
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
from joblib import dump
from transformers import AutoTokenizer, AutoModel
from src.helpers import get_embeddings, read_csv


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
N_COMPONENTS = 10


def main(config):
    tokenizer = AutoTokenizer.from_pretrained(config["base_tokenizer"])
    model = AutoModel.from_pretrained(config["base_model"])
    df = read_csv(config["data"])

    model.to(DEVICE)
    torch.cuda.empty_cache()
    embeddings = get_embeddings(model=model, tokenizer=tokenizer, sentences=df.pre_processed.values.tolist(), batch_size=BATCH_SIZE)


    np.save(os.path.join(config["save_folder"], f"before_pca_{config['type_data']}_embeddings.npy"), embeddings)

    pca = PCA(n_components=N_COMPONENTS)
    pca.fit(embeddings)
    embeddings = pca.transform(embeddings)
    np.save(os.path.join(config["save_folder"], f"after_pca_{config['type_data']}_embeddings.npy"), embeddings)

    dump(pca, os.path.join(config["save_folder"], "pca_model.joblib"))


if __name__ == '__main__':
    # python src/models/pca.py -bt xlm-roberta-base -bm models/roberta-fine-tuned-regression/checkpoint-7200  -d data/2023_09_28/train.csv -embeddings/regression -td train
    # python src/models/pca.py -bt xlm-roberta-base -bm final_models/mlm-fine-tuned-roberta/checkpoint-9000 -d data/2023_09_28/train.csv -sf embeddings/roberta_fine_tuned -td train
    ap = argparse.ArgumentParser()
    ap.add_argument('-bt', '--base_tokenizer', required=True,
                    help='base tokenizer for model')
    ap.add_argument('-bm', '--base_model', required=True,
                    help='base model')
    ap.add_argument('-d', '--data', required=True,
                    help='data to train the PCA')
    ap.add_argument('-sf', '--save_folder', required=True,
                    help='save_folder')
    ap.add_argument('-td', '--type_data', required=True,
                    help='type data, to be added in embeddings file names')
    args_main = vars(ap.parse_args())
    if not os.path.exists(args_main["save_folder"]):
        os.makedirs(args_main["save_folder"])
    with open(os.path.join(args_main["save_folder"], "config_pca.json"), "w") as f:
        json.dump(args_main, f, indent=4)
    main(config=args_main)
