
# -*- coding: utf-8 -*-
""" 
Extract embeddings from all models (custom fine-tuned and base from HuggingFace)
"""
import os
import click
import subprocess


def get_base_model(folder):
    """ Retrives model from the folder
    - if there is a *.pth file -> that's the model
    - if not -> taking the model from the last checkpoint """
    info = os.listdir(folder)
    if any(x.endswith(".pth") for x in info):  # Model stored as .pth file
        model = [x for x in info if x.endswith(".pth")][0]
        return os.path.join(folder, model)
    # config.json file in the folder
    checkpoints = [x for x in info if x.startswith("checkpoint-")]
    return os.path.join(folder, sorted(checkpoints)[-1])


@click.command()
@click.argument('folder_data')
@click.argument('save_folder')
@click.argument('save_models')
def main(folder_data, save_folder, save_models):
    models = [x for x in os.listdir(save_models) if os.path.isdir(os.path.join(save_models, x))]
    # Extracting embeddings from all fine-tuned models
    for model in models:
        save_folder = os.path.join(save_folder, model)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        base_model = get_base_model(folder=os.path.join(save_models, model))
        for type_data in ["eval", "train", "test"]:
            save_path = os.path.join(save_folder, f"{model}/{type_data}.npy")
            if not os.path.exists(save_path):
                command = f"python src/embeddings.py --base_model {base_model} --data data/2024_03_11/{type_data}.csv --save {save_path}"
                subprocess.call(command, shell=True)

    # Extracting embeddings from the base LLM
    for type_data in ["eval", "train", "test"]:
        save_path = os.path.join(save_folder, f"st_all_mpnet_base_v2/{type_data}.npy")
        if not os.path.exists(save_path):
            command = f"python src/embeddings.py --base_model sentence-transformers/all-mpnet-base-v2 --data data/2024_03_11/{type_data}.csv --save {save_path}"
            subprocess.call(command, shell=True)


if __name__ == '__main__':
    # python experiments/run_base_llm.py ./data/2024_03_11 final_embeddings final_models
    main()
