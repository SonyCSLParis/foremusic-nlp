# Lyrics for success: comparing stylometric and embedding features for song popularity prediction

This is the code that was submitted together with the paper "Lyrics for success: embedding features for song popularity prediction
", accepted to [NLP4MusA 2024](https://sites.google.com/view/nlp4musa-2024/home), co-located with [ISMIR'2024](https://ismir2024.ismir.net).

## Set Up a virtual environment

We strongly advise to set up a virtual experiments for these experiments.

```bash
pip install -r requirements.txt
```

You will also need to download additional resources from nltk in Python in your virtual environment.
```python
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
```

## Reproducibility

For more clarity, we describe the different scripts to run to reproduce our experiments in a separate [README](./experiments/README.md).

## Structure

Below an overview of the main content of the code, that is in the `src` folder:
* `configs`: configuration `.yaml` files for the regression layers
* `data_prep`: all scripts related to data preparation for model training
* `models`: all models
* `embeddings.py`: extract embeddings from a model
* `features.py`: stylometric features
* `helpers.py`: generic helpers

## Acknowledgments

If you use this work please cite the following paper:
```bib
@inproceedings{prevedello-etal-2024-lyrics,
    title = "Lyrics for Success: Embedding Features for Song Popularity Prediction",
    author = "Prevedello, Giulio  and
      Blin, Ines  and
      Monechi, Bernardo  and
      Ubaldi, Enrico",
    editor = "Kruspe, Anna  and
      Oramas, Sergio  and
      Epure, Elena V.  and
      Sordo, Mohamed  and
      Weck, Benno  and
      Doh, SeungHeon  and
      Won, Minz  and
      Manco, Ilaria  and
      Meseguer-Brocal, Gabriel",
    booktitle = "Proceedings of the 3rd Workshop on NLP for Music and Audio (NLP4MusA)",
    month = nov,
    year = "2024",
    address = "Oakland, USA",
    publisher = "Association for Computational Lingustics",
    url = "https://aclanthology.org/2024.nlp4musa-1.13/",
    pages = "75--80",
    abstract = "Accurate song success prediction is vital for the music industry, guiding promotion and label decisions. Early, accurate predictions are thus crucial for informed business actions. We investigated the predictive power of lyrics embedding features, alone and in combination with other stylometric features and various Spotify metadata (audio, platform, playlists, reactions). We compiled a dataset of 12,428 Spotify tracks and targeted popularity 15 days post-release. For the embeddings, we used a Large Language Model and compared different configurations. We found that integrating embeddings with other lyrics and audio features improved early-phase predictions, underscoring the importance of a comprehensive approach to success prediction."
}
```
