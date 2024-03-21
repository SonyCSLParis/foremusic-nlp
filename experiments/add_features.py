
import os
from src.features import FeatureExtractor
from src.helpers import read_csv

FOLDER_DATA = './data/2024_03_11'
FEAT_EXTRACTOR = FeatureExtractor()

for type_data in ["train", "eval", "test"]:
    df = read_csv(os.path.join(FOLDER_DATA, f"{type_data}.csv"))
    df = FEAT_EXTRACTOR(df=df)
    df.to_csv(os.path.join(FOLDER_DATA, f"{type_data}_with_feats.csv"))