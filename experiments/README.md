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

TO-DO-G: add info on the notebook