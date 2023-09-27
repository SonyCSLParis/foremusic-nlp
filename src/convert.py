# -*- coding: utf-8 -*-
"""
Convert pickled to pandas
"""
import argparse
import pickle
import pandas as pd

def convert_pkl_to_pd(pkl_path, csv_path):
    """ Convert pickle to pandas file """
    with open(pkl_path, 'rb') as openfile:
        data = pickle.load(openfile)
    track_ids = data.keys()
    df_out = pd.DataFrame({"track_id": list(track_ids),
                           "lyrics": [data[track_id] for track_id in track_ids]})
    df_out.to_csv(csv_path)


if __name__ == '__main__':
    # Command line examples (from repo directory)

    ap = argparse.ArgumentParser()
    ap.add_argument('-pkl', '--pickle', required=True,
                    help='input .pkl file to convert')
    ap.add_argument('-csv', '--csv', required=True,
                    help='output .csv file to save pandas dataframe')
    args_main = vars(ap.parse_args())

    convert_pkl_to_pd(pkl_path=args_main["pickle"], csv_path=args_main["csv"])

