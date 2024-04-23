# Lyrics for success: comparing stylometric and embedding features for song popularity prediction

This is the code that was submitted together with the paper at ISMIR 2024.

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

TO-DO-G: add the notebook in `src/models`

Below an overview of the main content of the code, that is in the `src` folder:
* `configs`: configuration `.yaml` files for the regression layers
* `data_prep`: all scripts related to data preparation for model training
* `models`: all models
* `embeddings.py`: extract embeddings from a model
* `features.py`: stylometric features
* `helpers.py`: generic helpers