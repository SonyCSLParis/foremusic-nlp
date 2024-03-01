# -*- coding: utf-8 -*-
""" 
Fine-tuned model for regression, adding two layers
- linear layer to reduce embedding dimension to 10
- regression
"""
import os
import torch
from datasets import Dataset
from types import NoneType
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments, PreTrainedModel
import torch
import torch.nn as nn
from src.helpers import get_data


class DimensionReductionRegressionHFModel(nn.Module):
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


class DimensionReductionRegressionModel:
    """ model """
    def __init__(self, config):
        self.keys_config = ["base_model", "learning_rate", "per_device_train_batch_size", "per_device_eval_batch_size", "num_train_epochs", "evaluation_strategy", "save_strategy", "save_total_limit", "logging_dir", "load_best_model_at_end", "output_dir"]
        check_config(keys=self.keys_config, config=config)
        self.config = config

        self.base_model = config["base_model"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self.model_auto = AutoModel.from_pretrained(self.base_model)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = DimensionReductionRegressionHFModel(self.model_auto)
        self.model.to(self.device)

        self.training_args = self.get_training_args
    
    def get_training_args():
        return TrainingArguments(
            output_dir=self.config["output_dir"],
            learning_rate=self.config["learning_rate"],
            per_device_train_batch_size=self.config["per_device_train_batch_size"],
            per_device_eval_batch_size=self.config["per_device_eval_batch_size"],
            num_train_epochs=self.config["num_train_epochs"],
            evaluation_strategy=self.config["evaluation_strategy"],
            save_strategy=self.config["save_strategy"],
            save_total_limit=self.config["save_total_limit"],
            logging_dir=self.config["logging_dir"],
            load_best_model_at_end=self.config["load_best_model_at_end"],
        )

    def train(self, dataset):
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=ds["train"],
            eval_dataset=ds["eval"],
        )
        trainer.train()
        torch.save(trainer.model, os.path.join(self.config["output_dir"], "model_10_dim_reg.pth"))

        # Step 5: Evaluation
        # eval_results = trainer.evaluate(eval_dataset=ds["test"])
        # print(eval_results)

@click.command()
@click.option("--train_path", help="Path to training dataset")
@click.option("--eval_path", help="Path to evaluation dataset")
@click.option("--config", help="Config path with parameters")
@click.option("--target", help="Target variable to predict")
def main(train_path, eval_path, config, target):
    """ Main to train 10-dim regression model """
    # Load model
    with open(config, "r") as f:
        config = yaml.safe_load(f)
    reg_model = DimensionReductionRegressionModel(config=config)

     # Load DS + Config
    train_ds = Dataset.from_csv(train_path)
    eval_ds = Dataset.from_csv(eval_path)
    dataset = get_data(reg_model.tokenizer, train_ds, eval_ds, target)

    return


if __name__ == '__main__':
    # python src/models/regression.py --train data/2023_09_28/train.csv --eval data/2023_09_28/eval.csv --config src/configs/base_10_dim_regression.yaml --target sp_pop_d15
    main()