# -*- coding: utf-8 -*-
"""
Splitting data (train/eval/test) into separate files
"""
import os
import argparse
import pandas as pd

def main(df_input, folder_output):
    """ Separating based on train/test set """
    train = df_input[df_input.train_eval_test == "train"]
    eval_ = df_input[df_input.train_eval_test == "eval"]
    test = df_input[df_input.train_eval_test == "test"]
    
    train.to_csv(os.path.join(folder_output, "train.csv"))
    eval_.to_csv(os.path.join(folder_output, "eval.csv"))
    test.to_csv(os.path.join(folder_output, "test.csv"))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', "--input", required=False,
                    help=".csv file with info")
    ap.add_argument('-o', "--output", required=False,
                    help="output folder for saving")
    args_main = vars(ap.parse_args())

    DF_INPUT = pd.read_csv(args_main["input"])
    DF_INPUT = DF_INPUT[[col for col in DF_INPUT.columns if col != "Unnamed: 0"]]

    main(df_input=DF_INPUT, folder_output=args_main["output"])
