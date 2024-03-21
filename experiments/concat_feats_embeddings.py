import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.helpers import read_csv

FOLDER_DATA = './data/2024_03_11'
EMBEDDINGS_PATH = "./final_embeddings"
EMBEDDINGS_FOLDER = [x for x in os.listdir(EMBEDDINGS_PATH) if os.path.isdir(os.path.join(EMBEDDINGS_PATH, x))]

def concat(df_path, embed_path):
    """ Concat embeddings into df """
    df = read_csv(df_path)
    df.reset_index(drop=True, inplace=True)

    embeddings = np.load(embed_path)
    df_embeddings = pd.DataFrame(embeddings, columns=[f'embedding_{i+1}' for i in range(embeddings.shape[1])])

    return pd.concat([df, df_embeddings], axis=1)

for embed_folder in tqdm(EMBEDDINGS_FOLDER):
    print(embed_folder)
    for type_data in ["train", "eval", "test"]:
        df_path = os.path.join(FOLDER_DATA, f"{type_data}_with_feats.csv")
        embed_path = os.path.join(EMBEDDINGS_PATH, embed_folder, f"{type_data}.npy")
        save_path =  os.path.join(EMBEDDINGS_PATH, embed_folder, "data.csv")
        df = concat(df_path=df_path, embed_path=embed_path)
        df.to_csv(save_path)
    
