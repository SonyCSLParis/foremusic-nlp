import yaml
import click
from datasets import Dataset
from src.helpers import get_data
from src.models.ten_dim_regression import DimensionReductionRegressionModel

TRAIN_DATA = "./data/2024_03_11/train.csv"
EVAL_DATA = "./data/2024_03_11/eval.csv"
TARGET = "sp_pop_d15"

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

def main(train_path, eval_path, config, shape, target):
    """ Main to train n-dim regression model """
    reg_model = DimensionReductionRegressionModel(config=config, output_embedding_shape=shape)

     # Load DS + Config
    train_ds = Dataset.from_csv(train_path)
    eval_ds = Dataset.from_csv(eval_path)
    dataset = get_data(reg_model.tokenizer, train_ds, eval_ds, target)

    reg_model.train(dataset=dataset)


def get_output_dir(base_model, shape):
    if base_model.startswith("./final_models"):
        llm = "ft_st_all_mpnet_base_v2"
    else:
        llm = "st_all_mpnet_base_v2"
    return f"./final_models/{llm}_dim_reduction_{shape}_ft_regression"



if __name__ == "__main__":
    for base_model in ["sentence-transformers/all-mpnet-base-v2", "./final_models/ft_st_all_mpnet_base_v2/checkpoint-948"]:
        for shape in [5, 10, 20, 30]:
            config = CONFIG_BASE
            config["base_model"] = base_model
            config["output_dir"] = get_output_dir(base_model, shape)
            try:
                main(train_path=TRAIN_DATA, eval_path=EVAL_DATA, config=config, shape=shape, target=TARGET)
            except:
                pass
