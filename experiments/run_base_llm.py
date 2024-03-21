import os
import subprocess

FOLDER_DATA = "./data/2024_03_11"
SAVE_FOLDER = "final_embeddings"
SAVE_MODELS = "./final_models"

MODELS = [x for x in os.listdir(SAVE_MODELS) if os.path.isdir(os.path.join(SAVE_MODELS, x))]

def get_base_model(folder):
    info = os.listdir(folder)
    if any(x.endswith(".pth") for x in info):  # Model stored as .pth file
        model = [x for x in info if x.endswith(".pth")][0]
        return os.path.join(folder, model)
    # config.json file in the folder
    checkpoints = [x for x in info if x.startswith("checkpoint-")]
    return os.path.join(folder, sorted(checkpoints)[-1])

for model in MODELS:
    save_folder = os.path.join(SAVE_FOLDER, model)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    base_model = get_base_model(folder=os.path.join(SAVE_MODELS, model))
    for type_data in ["eval", "train", "test"]:
        save_path = os.path.join(SAVE_FOLDER, f"{model}/{type_data}.npy")
        if not os.path.exists(save_path):
            command = f"python src/embeddings.py --base_model {base_model} --data data/2024_03_11/{type_data}.csv --save {save_path}"
            print(command)
            subprocess.call(command, shell=True)

for type_data in ["eval", "train", "test"]:
    save_path = os.path.join(SAVE_FOLDER, f"st_all_mpnet_base_v2/{type_data}.npy")
    if not os.path.exists(save_path):
        command = f"python src/embeddings.py --base_model sentence-transformers/all-mpnet-base-v2 --data data/2024_03_11/{type_data}.csv --save {save_path}"
        subprocess.call(command, shell=True)
