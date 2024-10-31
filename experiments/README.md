# ISMIR Experiments

This README contains additional information on the scripts to run for reproducibility of our results.


## 1. Data Preparation 

(target variables should be non null, preprocessing lyrics, filtering on language, dividing training/eval/test)
Assuming original data, stored as `table_lyrics_with_eval.csv`, is stored in `data/2024_03_11` folder

1. Preprocessing `lyrics` column and saving as new file
```bash
python src/data_prep/pre_process.py --path data/2024_03_11/table_lyrics_with_eval.csv -c lyrics -o data/2024_03_11/pre_processed.csv
```

2. Filtering + Adding language with transformer-based model
```bash
python src/data_prep/filter.py --input data/2024_03_11/pre_processed.csv --output data/2024_03_11/
```

3. Only keeping english-based models, and divide train/eval/test
```bash
python src/data_prep/divide_train_eval_test.py --input data/2024_03_11/filtered.csv --output data/2024_03_11/
```

4. Add stylometric features
```bash
python experiments/add_features.py ./data/2024_03_11
```

## 2. Fine-tune Language Model
We start with a generic language model, and we do domain adaptation with the lyrics

```bash
python src/models/fine_tune_llm.py --input data/2024_03_11/filtered.csv --folder final_models/ft_st_all_mpnet_base_v2
```

Base model (constant during the project): `sentence-transformers/all-mpnet-base-v2` 

At the end of this step, we have a fine-tuned LM (FT-LM), that embeds text with dimension 768

## 3. Train various models and store embeddings

1. `reg` models. Models with a regression layer only.
```bash
python src/models/regression.py --train_path data/2024_03_11/train.csv --eval_path data/2024_03_11/eval.csv --config src/configs/base_regression_sp.yaml --target sp_pop_d15
```
```bash
python src/models/regression.py --train_path data/2024_03_11/train.csv --eval_path data/2024_03_11/eval.csv --config src/configs/ft_regression_sp.yaml --target sp_pop_d15
```

2. `red-reg` models. Models with a dimensionality reduction and a regression layer.
```bash
python experiments/run_dim_red.py ./data/2024_03_11/train.csv ./data/2024_03_11/eval.csv sp_pop_d15 checkpoint-948
```

## 4. Extract embeddings from all models
```bash
python experiments/run_base_llm.py ./data/2024_03_11 final_embeddings final_models
```

## 5. Concatenate everything
```bash
python experiments/concat_feats_embeddings.py ./data/2024_03_11 ./final_embeddings
```

## 6. Final regression models 

The notebook src/models/train_test_fit.ipynb contains the code to train a LightGBM model for predicting song popularity 15 days post-release, leveraging various groups of song features combined with lyrics features, including both stylometric information and embeddings from large language models (LLMs). The key steps in the pipeline are:

* Loading data collected from Spotify and the generated embeddings, then merging these datasets.
* Setting the target variable as Spotify popularity at 15 days post-release and displaying it.
* Grouping features based on their availability over time.
* Applying UMAP for dimensionality reduction if embedding dimensions have not been previously reduced.
* Training the LightGBM model, then evaluating and visualizing predictions on the train and test sets along with feature importance scores.

To run the script, execute each cell in sequence as they appear in the notebook. For the second experiment, which uses the best-performing LLM model's lyrics embeddings, uncomment the relevant cells and comment out the analogous cells for the first experiment.
