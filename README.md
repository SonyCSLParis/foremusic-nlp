pip install torch torchvision torchaudio


language
pre-process text: remove [couplet 1], list 1. 2. 3., 2018\n <text>
embedding
(embedding + kg information?)
additional layer to reduce # of dimensions
training

Pipeline
1. Pre_process (text content)
2. Filter: extract language, only keep English
3. Divide train/eval/test
4. Train model

Hyperparameters
1. Pre-trained model


To predict on text from pretrained model
```python
import torch
from transformers import AutoTokenizer
from src.models import DimensionReductionRegressionModel

path_to_model = "/path/to/model"
model=torch.load(path)
model.eval()

path_tokenizer = "/path/to/tokenizer"
tokenizer = AutoTokenizer.from_pretrained(path_tokenizer)

input_test = "today I had lunch with friends"
inputs = tokenizer(input_text, return_tensors="pt")

with torch.no_grad():
    output = model(**inputs)

# to get last hidden state ~ embedding of dim 10
output.get("last_hidden_state")
```

## Notes 2024.03.01

Aim = see if latent representations of lyrics are good features to predict the popularity of a song
Two target variables: `yt_pop_d15` and `sp_pop_d15`

### 1st: Data Prepararation 

(target variables should be non null, preprocessing lyrics, filtering on language, dividing training/eval/test)
Assuming original data, stored as `table_lyrics_with_eval.csv`, is stored in `data/2024_03_01` folder

1. Preprocessing `lyrics` column and saving as new file
```bash
python src/data_prep/pre_process.py --path data/2024_03_01/table_lyrics_with_eval.csv -c lyrics -o data/2024_03_01/pre_processed.csv
```

2. Adding language with transformer-based model
```bash
```

3. Only keeping english-based models, and divide train/eval/test
```bash
```

### 2nd: Fine-tune Language Model
We start with a generic language model, and we do domain adaptation with the lyrics

```bash
python src/models/fine_tune_llm.py --input data/2024_03_01/filtered.csv --folder final_models/mlm-fine-tuned-roberta
```

Base model (constant during the project): `xlm-roberta-base` 

At the end of this step, we have a fine-tuned LM (FT-LM), that embeds text with dimension 768

### 3rd: Train various models and store embeddings

For each target variable there are four possible embedding types

1. 768-dimension embeddings from output of FT-LM
2. 10-dimension embeddings from FT-LM + PCA
(1. and 2. embeddings will be saved together)

For the below examples, the target variable is `sp_pop_d15`

Get the embeddings for train
```bash
python src/models/pca.py -bt xlm-roberta-base -bm final_models/mlm-fine-tuned-roberta/checkpoint-9000 -d data/2024_03_01/train.csv -sf embeddings/mlm-fine-tuned-roberta -td train
```

Store test embeddings
```bash
python src/embeddings.py --base_model final_models/mlm-fine-tuned-roberta/checkpoint-<CHANGE> --data data/2024_03_01/test.csv --save embeddings/mlm-fine-tuned-roberta/test_embeddings.npy --pca embeddings/mlm-fine-tuned-roberta/pca_model.joblib
```

3. 10-dimension embeddings from FT-LM + Regression + PCA 

Train the regression model 
```bash
python src/models/regression.py --train_path data/2024_03_01/train.csv --eval_path data/2024_03_01/eval.csv --config src/configs/base_regression.yaml --target sp_pop_d15
```

Train the PCA
```bash
python src/models/pca.py -bt xlm-roberta-base -bm final_models/roberta-fine-tuned-regression/checkpoint-<CHANGE> -d data/2024_03_01/train.csv -sf embeddings/roberta-fine-tuned-regression -td train
```

Store test embeddings
```bash
python src/embeddings.py --base_model final_models/roberta-fine-tuned-regression/checkpoint-<CHANGE> --data data/2024_03_01/test.csv --save embeddings/roberta-fine-tuned-regression/test_embeddings.npy --pca embeddings/roberta-fine-tuned-regression/pca_model.joblib
```

4. 10-dimension embeddings from FT-LM + Dimensionality Reduction + Regression + PCA

Train the regression model
```bash
python src/models/10_dim_regression.py --train_path data/2024_03_01/train.csv --eval_path data/2024_03_01/eval.csv --config src/configs/base_10_dim_regression.yaml --target sp_pop_d15
```

Train the PCA
```bash
python src/models/pca.py -bt xlm-roberta-base -bm final_models/reduction-regression-roberta/checkpoint-<CHANGE> -d data/2024_03_01/train.csv -sf embeddings/reduction-regression-roberta -td train
```

Store test embeddings
```bash
python src/embeddings.py --base_model final_models/reduction-regression-roberta/checkpoint-<CHANGE> --data data/2024_03_01/test.csv --save embeddings/reduction-regression-roberta/test_embeddings.npy --pca embeddings/reduction-regression-roberta/pca_model.joblib
```

### 4th: Load Embeddings and train DecisionTree Model

You can use the followings (adapt it depending on your path):

```python
from joblib import load
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

pca = load("pca_model.joblib")
data = "data/2023_09_28/train.csv"
target = "sp_pop_d15"
df = pd.read_csv(data)

X_train = np.load(embeddings.npy)
y_train = df[target].values.tolist()

decision_tree = DecisionTreeRegressor()
decision_tree.fit(X_train_10, y_train)
y_pred = decision_tree.predict(X_train_10)

mae = mean_absolute_error(y_train, y_pred)
mse = mean_squared_error(y_train, y_pred)
r2 = r2_score(y_train, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R^2 (Coefficient of Determination):", r2)
```

