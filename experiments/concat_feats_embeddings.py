# -*- coding: utf-8 -*-
""" 
Concatenate all features except embeddings with embedding-based features
"""
import os
import click
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.helpers import read_csv


def concat(df_path, embed_path):
    """ Concat embeddings into df """
    df = read_csv(df_path)
    df.reset_index(drop=True, inplace=True)

    embeddings = np.load(embed_path)
    df_embeddings = pd.DataFrame(embeddings, columns=[f'embedding_{i+1}' for i in range(embeddings.shape[1])])

    return pd.concat([df, df_embeddings], axis=1)


@click.command()
@click.argument('folder_data')
@click.argument('embeddings_path')
def main(folder_data, embeddings_path):
    embeddings_folder = [x for x in os.listdir(embeddings_path) if os.path.isdir(os.path.join(embeddings_path, x))]

    for embed_folder in tqdm(embeddings_folder):
        for type_data in ["train", "eval", "test"]:
            df_path = os.path.join(folder_data, f"{type_data}_with_feats.csv")
            embed_path = os.path.join(embeddings_path, embed_folder, f"{type_data}.npy")
            save_path =  os.path.join(embeddings_path, embed_folder, f"data_{type_data}.csv")
            df = concat(df_path=df_path, embed_path=embed_path)
            df.to_csv(save_path)


if __name__ == '__main__':
    # python experiments/concat_feats_embeddings.py ./data/2024_03_11 ./final_embeddings
    main()
    
