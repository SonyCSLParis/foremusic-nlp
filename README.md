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