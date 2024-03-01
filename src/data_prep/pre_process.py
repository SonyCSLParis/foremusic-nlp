# -*- coding: utf-8 -*-
"""
pre-processing 
"""
import re
import argparse
import pandas as pd
from tqdm import tqdm


class PreProcessor:
    """ Pre-process lyrics """
    def __init__(self):
        # regexes to remove parts that indicate which part (singer/chorus, etc) is singing
        self.regex_parts = re.compile("\\[.+\\]", re.MULTILINE)
        # listing
        self.listing_1 = re.compile("(\\d+\\..+(\\n|))+", re.MULTILINE)
        self.listing_2 = re.compile(".+\s-\s.+\n", re.MULTILINE)
        # dates at the end
        self.dates_end = re.compile(".+\\s\\((January|February|March|April|May|June|July|August|September|October|November|December)\\s\\d{1,2},\\s\\d{4}\\)", re.MULTILINE)
        # date list: 2020\n list of songs (repeated)
        self.dates_list = re.compile("(\\d{4}\\s(.+\\s)+)+", re.MULTILINE)
        # playlist
        self.playlist = re.compile(".+(of|from) the Playlist", re.MULTILINE)
        # nb Contributor
        self.contributor = re.compile("\d+ Contributor.+", re.MULTILINE)


        self.replace = [("↗", "")]
        self.pipeline = [
            self.remove_parts,
            lambda text: self.remove_simple(text, self.listing_1),
            lambda text: self.remove_simple(text, self.listing_2),
            lambda text: self.remove_simple(text, self.dates_end),
            lambda text: self.remove_simple(text, self.dates_list),
            lambda text: self.remove_simple(text, self.playlist),
            lambda text: self.remove_simple(text, self.contributor)
        ]
        self.matches = []

    def remove_parts(self, text):
        """ Remove content between brackets: [] """
        self.matches = re.findall(self.regex_parts, text)
        text = re.sub(self.regex_parts, '', text)
        return text

    def remove_simple(self, text, pattern):
        """ Remove all patterns from text """
        return re.sub(pattern, '', text)

    def __call__(self, text):
        for (old, new) in self.replace:
            text = text.replace(old, new)

        for func in self.pipeline:
            text = func(text.strip()).strip()

        return text


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-p', "--path", required=False,
                    help=".csv file with info")
    ap.add_argument('-c', "--col", required=False,
                    help="column to pre-process")
    ap.add_argument('-o', "--output", required=False,
                    help="output file for saving")
    args_main = vars(ap.parse_args())

    DF_ = pd.read_csv(args_main["path"])
    print(DF_.shape)
    DF_ = DF_[(~DF_.yt_pop_d15.isna()) & (~DF_.sp_pop_d15.isna())]
    DF_ = DF_.fillna("")
    print(DF_.shape)
    PRE_PROCESSOR = PreProcessor()
    tqdm.pandas()
    DF_["pre_processed"] = DF_[args_main["col"]].progress_apply(PRE_PROCESSOR)
    DF_.to_csv(args_main["output"])
