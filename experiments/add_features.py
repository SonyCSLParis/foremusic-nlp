# -*- coding: utf-8 -*-
""" 
Add stylometric features to the train/eval/test data
"""
import os
import click
from src.features import FeatureExtractor
from src.helpers import read_csv

FEAT_EXTRACTOR = FeatureExtractor()

@click.command()
@click.argument('folder_data')
def main(folder_data):
    for type_data in ["train", "eval", "test"]:
        df = read_csv(os.path.join(folder_data, f"{type_data}.csv"))
        df = FEAT_EXTRACTOR(df=df)
        df.to_csv(os.path.join(folder_data, f"{type_data}_with_feats.csv"))


if __name__ == '__main__':
    # python experiments/add_features.py ./data/2024_03_11
    main()