# -*- coding: utf-8 -*-
"""
Filtering based on following criteria
- lyrics and score should be non-empty
- removing if not english
"""
import os
import psutil
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

def get_label(res):
    """ Classifying language """
    if res:
        return res["label"]
    return "unknown"

def main(df_input, f_write):
    """ Main filtering """
    print(f"{df_input.shape[0]}: original")
    f_write.write(f"{df_input.shape[0]}: original\n")
    df_input = df_input.fillna("")
    non_empty_col = ["pre_processed", "yt_pop_d15", "sp_pop_d15"]
    for col in non_empty_col:
        df_input = df_input[df_input[col] != ""]
        print(f"{df_input.shape[0]}: after removing empty {col}")
        f_write.write(f"{df_input.shape[0]}: after removing empty {col}\n")
    
    # Checking if lyrics are unique per track_id > 
    # if not, assuming the lyrics are not well extracted
    df_input.pre_processed = df_input.pre_processed.apply(lambda x: x.replace("Embed", ""))
    grouped = df_input.groupby("pre_processed").agg({"track_id": "count"}).sort_values(by="track_id", ascending=False)
    grouped = grouped.reset_index().rename(columns={"track_id": "same_lyrics_count"})
    df_merged = pd.merge(df_input, grouped, on="pre_processed", how="left")
    df_filter = df_merged[df_merged.same_lyrics_count == 1].fillna("")
    print(f"{df_filter.shape[0]}: after removing duplicated lyrics\n")
    f_write.write(f"{df_filter.shape[0]}: after removing duplicated lyrics\n")

    df_filter = df_filter[df_filter.pre_processed != ""]
    df_filter["lyrics_len"] = df_filter['pre_processed'].apply(len)
    df_filter = df_filter[df_filter.lyrics_len >= 500]
    print(f"{df_filter.shape[0]}: after removing lyrics with less than 500 characters\n")
    f_write.write(f"{df_filter.shape[0]}: after removing lyrics with less than 500 characters\n")
    
    return df_filter

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', "--input", required=True,
                    help=".csv file with info")
    ap.add_argument('-o', "--output", required=True,
                    help="output folder for saving")
    args_main = vars(ap.parse_args())

    DF_INPUT = pd.read_csv(args_main["input"])
    DF_INPUT = DF_INPUT[[col for col in DF_INPUT.columns if col != "Unnamed: 0"]]

    f_output = open(os.path.join(args_main["output"], "stats.txt"), "w+", encoding="utf-8")
    DF_INPUT = main(df_input=DF_INPUT, f_write=f_output)

    INPUT_LIST = list(DF_INPUT.lyrics.values)

    RESULTS = []

    BATCH_SIZE = 100
    # Iterate over the input_texts in batches
    for i in tqdm(range(0, len(INPUT_LIST), BATCH_SIZE), desc="Processing Batches"):
        batch = INPUT_LIST[i:i + BATCH_SIZE]
        batch_results = CLASSIFIER(batch, batch_size=100, truncation=True, padding="max_length", max_length=512)
        RESULTS.extend([get_label(x) for x in batch_results])

    DF_INPUT["language"] = RESULTS

    print(f"Original size: {DF_INPUT.shape[0]}")
    f_output.write(f"Original size: {DF_INPUT.shape[0]}\n")
    DF_INPUT.to_csv(os.path.join(args_main["output"], "language.csv"))
    print(f"Filtered size: {DF_INPUT[DF_INPUT.language.isin(['en-US', 'en-UK'])].shape[0]}")
    f_output.write(f"Filtered size: {DF_INPUT[DF_INPUT.language.isin(['en-US', 'en-UK'])].shape[0]}")
    DF_INPUT[DF_INPUT.language.isin(["en-US", "en-UK"])].to_csv(os.path.join(args_main["output"], "filtered.csv"))
    f_output.close()
