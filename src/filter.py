# -*- coding: utf-8 -*-
"""
Filtering based on following criteria
- lyrics and score should be non-empty
- removing if not english
"""
import os
import argparse
import multiprocessing as mp
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    TextClassificationPipeline

MODEL_NAME = 'qanastek/51-languages-classifier'
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
CLASSIFIER = TextClassificationPipeline(model=MODEL, tokenizer=TOKENIZER)

def mp_func(text):
    """ Classifying language """
    res = CLASSIFIER(text, truncation=True, max_length=512)
    if res:
        return res[0]["label"]
    return "unknown"

def main(df_input, f_write):
    """ Main filtering """
    print(f"{df_input.shape[0]}: original")
    f_write.write(f"{df_input.shape[0]}: original\n")
    df_input = df_input.fillna("")
    non_empty_col = ["lyrics", "yt_pop_d15"]
    for col in non_empty_col:
        df_input = df_input[df_input[col] != ""]
        print(f"{df_input.shape[0]}: after removing empty {col}")
        f_write.write(f"{df_input.shape[0]}: after removing empty {col}\n")

    return df_input

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', "--input", required=False,
                    help=".csv file with info")
    ap.add_argument('-o', "--output", required=False,
                    help="output folder for saving")
    args_main = vars(ap.parse_args())

    DF_INPUT = pd.read_csv(args_main["input"])
    DF_INPUT = DF_INPUT[[col for col in DF_INPUT.columns if col != "Unnamed: 0"]]

    f_output = open(os.path.join(args_main["output"], "stats.txt"), "w+", encoding="utf-8")
    DF_INPUT = main(df_input=DF_INPUT, f_write=f_output)

    with mp.Pool(processes=8) as pool:
        results = []
        for result in tqdm(pool.map(mp_func, DF_INPUT.lyrics.values),
                           total=DF_INPUT.shape[0]):
            results.append(result)

        pool.close()
        pool.join()
    
    DF_INPUT["language"] = results

    print(f"Original size: {DF_INPUT.shape[0]}")
    f_output.write(f"Original size: {DF_INPUT.shape[0]}")
    DF_INPUT.to_csv(os.path.join(args_main["output"], "language.csv"))
    print(f"Filtered size: {DF_INPUT[DF_INPUT.language.isin(['en-US', 'en-UK'])].shape[0]}")
    f_output.write(f"Filtered size: {DF_INPUT[DF_INPUT.language.isin(['en-US', 'en-UK'])].shape[0]}")
    DF_INPUT[DF_INPUT.language.isin(["en-US", "en-UK"])].to_csv(os.path.join(args_main["output"], "filtered.csv"))
    f_output.close()
