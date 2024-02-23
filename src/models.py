# -*- coding: utf-8 -*-
""" 
"""
import torch
from datasets import Dataset
from types import NoneType
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments, PreTrainedModel
import torch
import torch.nn as nn

def preprocess_function(examples):
    """ Pre-processing text and score """
    label = examples["yt_pop_d15"] 
    examples = tokenizer(examples["pre_processed"], truncation=True, padding="max_length", max_length=256)

    # Change this to real number -> removed because already float
    examples["label"] = float(label)
    return examples

class DimensionReductionRegressionModel(nn.Module):
    def __init__(self, base_model, output_embedding_shape: int = 10):
        super(DimensionReductionRegressionModel, self).__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(0.1)  # You can adjust dropout rate if needed
        self.dim_reduction_layer = nn.Linear(base_model.config.hidden_size, output_embedding_shape)
        self.regression_layer = nn.Linear(output_embedding_shape, 1)

    def forward(self, input_ids, attention_mask, labels=None):
        input_ids.to(self.base_model.device)
        attention_mask = attention_mask.to(self.base_model.device)

        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        dim_reduced_output = self.dim_reduction_layer(pooled_output)

        logits = self.regression_layer(dim_reduced_output)

        if not isinstance(labels, NoneType):
            labels = labels.to(self.base_model.device)
            loss = nn.MSELoss()(logits.flatten(), labels.flatten())
            return {
                "loss": loss, "logits": logits,
                "last_hidden_state": dim_reduced_output, "hidden_states": outputs.hidden_states,
                "attentions": outputs.attentions
            }
            # return loss
        # return logits
        return {
                "logits": logits,
                "last_hidden_state": dim_reduced_output, "hidden_states": outputs.hidden_states,
                "attentions": outputs.attentions
            }

if __name__ == '__main__':
    # Load DS
    train_ds = Dataset.from_csv("data/2023_09_28/train.csv")
    eval_ds = Dataset.from_csv("data/2023_09_28/eval.csv")
    test_ds = Dataset.from_csv("data/2023_09_28/test.csv")
    ds = {"train": train_ds.select(range(1000)), "eval": eval_ds.select(range(100)), "test": test_ds.select(range(100))}
    # ds = {"train": train_ds, "eval": eval_ds, "test": test_ds}

    BASE_MODEL = "./models/mlm-fine-tuned-roberta/checkpoint-9000"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModel.from_pretrained(BASE_MODEL)

    remove_columns = ['Unnamed: 0', 'track_id', 'start_date', 'release_date', 'artist_id', 'artist_name', 'album_id', 'album_name', 'ytv_ref', 'ytc_id', 'db_country', 'lyrics', 'train_eval_test']
    for split in ds:
        ds[split] = ds[split].map(preprocess_function, remove_columns=remove_columns)



    model = DimensionReductionRegressionModel(model)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Step 4: Fine-tuning the Model
    training_args = TrainingArguments(
        output_dir="./models/reduction-regression-roberta",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit = 2,
        logging_dir="./logs",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["eval"],
    )

    trainer.train()
    output_dir = "./models/reduction-regression-roberta"
    trainer.save_model(output_dir)
    torch.save(trainer.model, "./models/reduction-regression-roberta/model.pth")

    # Step 5: Evaluation
    eval_results = trainer.evaluate(eval_dataset=ds["test"])
    print(eval_results)
