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
import umap
from sklearn.decomposition import PCA
from joblib import dump
from transformers import AutoTokenizer, AutoModel
from src.helpers import get_embeddings, read_csv


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
N_COMPONENTS = 10


def main(config):
    models = ["pca", "umap"]
    if config.get("type_model") not in models:
        raise ValueError(f"Parameter `type_model` must be within {models}")
    tokenizer = AutoTokenizer.from_pretrained(config["base_tokenizer"])
    model = AutoModel.from_pretrained(config["base_model"])
    df = read_csv(config["data"])
    type_model = config.get("type_model")

    model.to(DEVICE)
    torch.cuda.empty_cache()
    embeddings = get_embeddings(model=model, tokenizer=tokenizer, sentences=df.pre_processed.values.tolist(), batch_size=BATCH_SIZE)


    np.save(os.path.join(config["save_folder"], f"before_{type_model}_{config['type_data']}_embeddings.npy"), embeddings)

    if type_model == "pca":
        dim_red_model = PCA(n_components=N_COMPONENTS)
        dim_red_model.fit(embeddings)
        embeddings = dim_red_model.transform(embeddings)
    else:  # type_model == "umap"
        dim_red_model = umap.UMAP(n_components=N_COMPONENTS)
        embeddings = dim_red_model.fit_transform(embeddings)

    np.save(os.path.join(config["save_folder"], f"after_{type_model}_{config['type_data']}_embeddings.npy"), embeddings)

    dump(dim_red_model, os.path.join(config["save_folder"], f"{type_model}_model.joblib"))


if __name__ == '__main__':
    # python src/models/dimensionality_reduction.py -tm pca -bt xlm-roberta-base -bm models/roberta-fine-tuned-regression/checkpoint-7200  -d data/2023_09_28/train.csv -embeddings/regression -td train
    # python src/models/dimensionality_reduction.py -tm pca -bt xlm-roberta-base -bm final_models/mlm-fine-tuned-roberta/checkpoint-9000 -d data/2023_09_28/train.csv -sf embeddings/roberta_fine_tuned -td train
    ap = argparse.ArgumentParser()
    ap.add_argument('-tm', '--type_model', required=True,
                    help='Model to use for dimensionality reduction')
    ap.add_argument('-bt', '--base_tokenizer', required=True,
                    help='base tokenizer for model')
    ap.add_argument('-bm', '--base_model', required=True,
                    help='base model')
    ap.add_argument('-d', '--data', required=True,
                    help='data to train the dimensionality reduction model')
    ap.add_argument('-sf', '--save_folder', required=True,
                    help='save_folder')
    ap.add_argument('-td', '--type_data', required=True,
                    help='type data, to be added in embeddings file names')
    args_main = vars(ap.parse_args())
    if not os.path.exists(args_main["save_folder"]):
        os.makedirs(args_main["save_folder"])
    with open(os.path.join(args_main["save_folder"], f"config_{args_main['type_model']}.json"), "w") as f:
        json.dump(args_main, f, indent=4)
    main(config=args_main)
