# -*- coding: utf-8 -*-
""" 
Extracting (traditional) features from lyrics, such as token count, etc

Re-implementation from the following paper: ALF-200k: Towards Extensive Multimodal Analyses of Music Tracks and Playlists
"""
import click
from tqdm import tqdm
import string
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from src.helpers import read_csv

def preprocess_text(text):
    """ Tokenizing and removing punctuation+stopwords"""
    tokens = word_tokenize(text.lower())
    return tokens

class LexicalFeatureExtractor:
    """ Extracting Lexical Feature """
    def __init__(self):
        """ Still missing:
        - Bag-of-Words (must be done on the whole dataset)
        - words/lines/chars per min (we do not have such information) """
        self.stop_words = set(stopwords.words('english'))

    
    def __call__(self, text):
        """ Retrieving all lexical features:
        - token count
        - unique token ratio
        - avg token length
        - repeated token ratio: 1 - unique token ratio
        - hapax dis-/tris-/legomenon: words that appear exactly once/twice/three times in the text
        - unique tokens/line
        - avg tokens/line
        - line counts
        - punctuation and digit ratios
        - stop word ratio
        - stop words / line """
        lines = sent_tokenize(text)
        tokens = preprocess_text(text=text)

        # Variables to be re-used throughout this function
        token_count = len(tokens)
        unique_tokens = set(tokens)
        unique_token_ratio = len(unique_tokens)/token_count
        bow = FreqDist(tokens)
        unique_tokens_per_line = [len(set(preprocess_text(line))) for line in lines]
        line_count = len(lines)
        stop_words_count = sum(1 for token in tokens if token in self.stop_words)
        stop_words_per_line = [sum(1 for token in word_tokenize(line) if token in self.stop_words) for line in lines]

        return {
            "token_count": token_count,
            "unique_token_ratio": unique_token_ratio,
            "avg_token_length": sum(len(token) for token in tokens) / token_count if token_count else 0,
            "repeated_token_ratio": 1 - unique_token_ratio,
            "hapax_dislegomenon": sum(1 for word, freq in bow.items() if freq == 1),
            "hapax_trislegomenon": sum(1 for word, freq in bow.items() if freq == 2),
            "hapax_legomenon": sum(1 for word, freq in bow.items() if freq == 3),
            "avg_unique_tokens_per_line": sum(unique_tokens_per_line) / line_count if line_count else 0,
            "avg_tokens_per_line": sum(len(word_tokenize(line)) for line in lines) / len(lines) if lines else 0,
            "line_count": line_count,
            "punctuation_ratio": sum(1 for token in tokens if token in string.punctuation) / token_count if token_count else 0,
            "digit_ratio": sum(1 for token in tokens if token.isdigit()) /token_count if token_count else 0,
            "stop_words_ratio": stop_words_count / token_count if token_count else 0,
            "avg_stop_words_per_line": sum(stop_words_per_line) / len(lines) if lines else 0
        }


class LinguisticFeatureExtractor:
    """ Extracting linguistic features """
    def __init__(self):
        """ Still missing
        - uncommon words ratios (how do you define uncommon?)
        - slang words ratio (how do you define slang?)
        - rhyme analyzer (cf. paper, needs to define your rhymes)
        - echoisms (same, need to define your patterns)
        - repetitive structures (eg chorus, refrain) (check further data)
        """
        pass

    @staticmethod
    def get_lemma_ratio(tokens):
        lemmas = set()
        for word in tokens:
            lemma = wn.morphy(word)
            if lemma is not None:
                lemmas.add(lemma)
        return len(lemmas) / len(tokens)
    
    @staticmethod
    def identify_repetitive_structures(text):
        # Define patterns for repetitive structures (e.g., choruses, refrains)
        patterns = [r'chorus', r'refrain']  # Example patterns, customize as needed
        repetitive_structures = []
        for pattern in patterns:
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            repetitive_structures.extend(matches)
        return len(repetitive_structures)
    
    def __call__(self, text):
        tokens = preprocess_text(text=text)

        return {
            "lemma_ratio": self.get_lemma_ratio(tokens) if tokens else 0
        }


class SemanticFeatureExtractor:
    """ Extracting semantic features """
    def __init__(self):
        """ Remaining (necessitates additional knowledge) 
        - Regressive imagery (RI) conceptual thought features
        - RI emotion features
        - RI primordial thought features
        - SentiStrength scores
        - AFINN scores
        - Opinion Lexicon scores
        """
        self.sid = SentimentIntensityAnalyzer()
    
    def __call__(self, text):
        return self.sid.polarity_scores(text)


class SyntacticFeatureExtractor:
    """ Syntactic feature extractor """
    def __init__(self):
        """ Remaining:
        - POS bag of words (should be on whole text)
        - POS frequencies
        - text chunks
        """
        pass
    
    def __call__(self, text):
        tokens = word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)

        past_tense_verbs = sum(1 for word, pos in pos_tags if pos == 'VBD')
        verbs = sum(1 for word, pos in pos_tags if pos.startswith('VB'))
        
        return {
            "pronoun_freq": sum(1 for word, pos in pos_tags if pos == 'PRP') / len(tokens) if tokens else 0,
            "past_tense_ratio": past_tense_verbs / verbs if verbs else 0
        }


class FeatureExtractor:
    """ All feature extractor """
    def __init__(self):
        self.lexical_feat = LexicalFeatureExtractor()
        self.linguistic_feat = LinguisticFeatureExtractor()
        self.semantic_feat = SemanticFeatureExtractor()
        self.syntactic_feat = SyntacticFeatureExtractor()

        self.features = [
            ("lexical", LexicalFeatureExtractor()),
            ("linguistic", LinguisticFeatureExtractor()),
            ("semantic", SemanticFeatureExtractor()),
            ("syntacric", SyntacticFeatureExtractor())
        ]
    
    def extract_features(self, row):
        for (name, extractor) in self.features:
            curr_feat = extractor(row["pre_processed"])
            for k, val in curr_feat.items():
                row[f"{name}_{k}"] = val
        return row
    
    def __call__(self, df):
        tqdm.pandas()
        df = df.progress_apply(self.extract_features, axis=1)
        return df


@click.command()
@click.argument('input_data')
@click.argument('output_data')
def main(input_data, output_data):
    """ Main """
    df = read_csv(input_data)
    feat_extractor = FeatureExtractor()
    df = feat_extractor(df)
    df.to_csv(output_data)


if __name__ == '__main__':
    # python src/features.py data/2024_03_01/filtered.csv ./filtered_with_feats.csv
    main()