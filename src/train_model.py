# -*- coding: utf-8 -*-
""" 
"""
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

#  Metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Load DS
train_ds = Dataset.from_csv("data/2023_09_28/train.csv")
eval_ds = Dataset.from_csv("data/2023_09_28/eval.csv")

# Set Up
BASE_MODEL = "xlm-roberta-base"
LEARNING_RATE = 2e-5
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 20

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=1)

# Prepare Datasets
# ds = {"train": train_ds.select(range(100)), "eval": eval_ds.select(range(10))}
ds = {"train": train_ds, "eval": eval_ds}

def preprocess_function(examples):
    """ Pre-processing text and score """
    label = examples["yt_pop_d15"] 
    examples = tokenizer(examples["pre_processed"], truncation=True, padding="max_length", max_length=256)

    # Change this to real number -> removed because already float
    examples["label"] = float(label)
    return examples

remove_columns = ['Unnamed: 0', 'track_id', 'start_date', 'release_date', 'artist_id', 'artist_name', 'album_id', 'album_name', 'ytv_ref', 'ytc_id', 'db_country', 'lyrics', 'train_eval_test']
for split in ds:
    ds[split] = ds[split].map(preprocess_function, remove_columns=remove_columns)

def compute_metrics_for_regression(eval_pred):
    logits, labels = eval_pred
    labels = labels.reshape(-1, 1)

    mse = mean_squared_error(labels, logits)
    mae = mean_absolute_error(labels, logits)
    r2 = r2_score(labels, logits)
    single_squared_errors = ((logits - labels).flatten()**2).tolist()
    
    return {"mse": mse, "mae": mae, "r2": r2}

training_args = TrainingArguments(
    output_dir="./models/roberta-fine-tuned-regression",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    metric_for_best_model="mse",
    load_best_model_at_end=True,
    weight_decay=0.01,
)

# Loss Function

class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs[0][:, 0]
        loss = torch.nn.functional.mse_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss


# Training
trainer = RegressionTrainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["eval"],
    compute_metrics=compute_metrics_for_regression,
)

trainer.train()
