# -*- coding: utf-8 -*-
""" 
Fine-tuned model for regression, on the n-shaped embeddings
"""
import yaml
import click
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

#  Metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from src.helpers import get_data, check_config


def compute_metrics_for_regression(eval_pred):
    logits, labels = eval_pred
    labels = labels.reshape(-1, 1)

    mse = mean_squared_error(labels, logits)
    mae = mean_absolute_error(labels, logits)
    r2 = r2_score(labels, logits)
    single_squared_errors = ((logits - labels).flatten()**2).tolist()
    
    return {"mse": mse, "mae": mae, "r2": r2}


# Loss Function

class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs[0][:, 0]
        loss = torch.nn.functional.mse_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss



class RegressionModel:
    """ Simple regresion model """
    def __init__(self, config):
        self.keys_config = ["base_model", "learning_rate", "max_length", "batch_size", "epochs", "evaluation_strategy", "save_strategy", "save_total_limit", "metric_for_best_model", "load_best_model_at_end", "weight_decay", "output_dir"]
        check_config(keys=self.keys_config, config=config)
        self.config = config

        self.base_model = self.config["base_model"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.base_model, num_labels=1)

        self.training_args = self.get_training_args()

    
    def get_training_args():
        return TrainingArguments(
            output_dir=self.config["output_dir"],
            learning_rate=self.config["learning_rate"],
            per_device_train_batch_size=self.config["batch_size"],
            per_device_eval_batch_size=self.config["batch_size"],
            num_train_epochs=self.config["epochs"],
            evaluation_strategy=self.config["evaluation_strategy"],
            save_strategy=self.config["save_strategy"],
            save_total_limit=self.config["save_total_limit"],
            metric_for_best_model=self.config["metric_for_best_model"],
            load_best_model_at_end=self.config["load_best_model_at_end"],
            weight_decay=self.config["weight_decay"],
        )
    
    def train(self, dataset):
        trainer = RegressionTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["eval"],
            compute_metrics=compute_metrics_for_regression,
        )
        trainer.train()

@click.command()
@click.option("--train_path", help="Path to training dataset")
@click.option("--eval_path", help="Path to evaluation dataset")
@click.option("--config", help="Config path with parameters")
@click.option("--target", help="Target variable to predict")
def main(train_path, eval_path, config, target):
    """ Main to train regression model """
    # Load model
    with open(config, "r") as f:
        config = yaml.safe_load(f)
    reg_model = RegressionModel(config=config)

    # Load DS
    train_ds = Dataset.from_csv(train_path)
    eval_ds = Dataset.from_csv(eval_path)
    dataset = get_data(reg_model.tokenizer, train_ds, eval_ds, target)

    reg_model.train(dataset=dataset)


if __name__ == '__main__':
    # python src/models/regression.py --train data/2023_09_28/train.csv --eval data/2023_09_28/eval.csv --config src/configs/base_regression.yaml
    main()
