# -*- coding: utf-8 -*-
""" 
Training and storing the models with a dimensionality reduction layer
"""
import yaml
import click
from datasets import Dataset
from src.helpers import get_data
from src.models.ten_dim_regression import DimensionReductionRegressionModel

CONFIG_BASE = {
    "learning_rate": 0.00002,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "num_train_epochs": 20,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "save_total_limit": 2,
    "logging_dir": "./logs",
    "load_best_model_at_end": True
}


def get_output_dir(base_model, shape):
    """ Output save folder to save models """
    if base_model.startswith("./final_models"):
        llm = "ft_st_all_mpnet_base_v2"
    else:
        llm = "st_all_mpnet_base_v2"
    return f"./final_models/{llm}_dim_reduction_{shape}_ft_regression"


@click.command()
@click.argument('train_data')
@click.argument('eval_data')
@click.argument('target')
def main(train_data, eval_data, target, checkpoint):
    # two base models: base model, fine-tuned model
    for base_model in ["sentence-transformers/all-mpnet-base-v2", f"./final_models/ft_st_all_mpnet_base_v2/{checkpoint}"]:
        # 4 embeddings size
        for shape in [5, 10, 20, 30]:
            config = CONFIG_BASE
            config["base_model"] = base_model
            config["output_dir"] = get_output_dir(base_model, shape)
            try:
                # Train n-dim regression model
                reg_model = DimensionReductionRegressionModel(config=config, output_embedding_shape=shape)

                # Load DS + Config
                train_ds = Dataset.from_csv(train_data)
                eval_ds = Dataset.from_csv(eval_data)
                dataset = get_data(reg_model.tokenizer, train_ds, eval_ds, target)

                reg_model.train(dataset=dataset)
            except:
                pass


if __name__ == "__main__":
    # python experiments/run_dim_red.py ./data/2024_03_11/train.csv ./data/2024_03_11/eval.csv sp_pop_d15 checkpoint-948
    main()
