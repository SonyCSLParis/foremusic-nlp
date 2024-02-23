# -*- coding: utf-8 -*-
"""
Fine-tuning pre-trained LLM on lyrics data
https://huggingface.co/learn/nlp-course/chapter7/3
NB: load tokenizer before model

export CUDA_VISIBLE_DEVICES=0 (GPU imbalance)
"""
import os
import collections
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling
from transformers import default_data_collator


MODEL_NAME = "xlm-roberta-base"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
MODEL = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
DATA_COLLATOR = DataCollatorForLanguageModeling(tokenizer=TOKENIZER, mlm_probability=0.15)
WWM_PROBABILITY = 0.2

def select_data(df_input):
    """ Filtering input data
    1. language: English
    2. only train/eval data """
    df_input = df_input[
        (df_input.language.isin(["en-US", "en-UK"])) & \
        (df_input.train_eval_test.isin(["train", "eval"]))]
    return df_input[[col for col in df_input.columns if col != "Unnamed: 0"]]


def split_train_test(dataset):
    """ Splitting """
    return dataset.shuffle(seed=23).train_test_split(test_size=0.1, seed=23)


def tokenize_function(examples):
    # result = TOKENIZER(examples["pre_processed"], truncation=True, padding="max_length", max_length=256)
    result = TOKENIZER(examples["pre_processed"])
    if TOKENIZER.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result


def group_texts(examples, chunk_size=128):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result


def whole_word_masking_data_collator(features):
    for feature in features:
        word_ids = feature.pop("word_ids")

        # Create a map between words and corresponding token indices
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # Randomly mask words
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id
        feature["labels"] = new_labels

    return default_data_collator(features)


def main(input, folder):
    """ Main for this script: 
    1. Filter data for fine-tuning
    2. Load datasets: train/eval/test """

    # Save data for MLM fine-tuning
    df_input = select_data(df_input=pd.read_csv(args_main["input"]))
    input_data_fine_tuning = os.path.join(args_main["folder"], "input_data_fine_tuning.csv")
    df_input.to_csv(input_data_fine_tuning)
    remove_columns = list(set(list(df_input.columns) + ["Unnamed: 0"])) 

    # Load dataset
    dataset = Dataset.from_csv(input_data_fine_tuning)
    ds = split_train_test(dataset=dataset)

    # Prep for training: tokenizing, chunking
    tokenized_datasets = ds.map(
        tokenize_function, batched=True, remove_columns=remove_columns)
    print(TOKENIZER.model_max_length)
    print(tokenized_datasets)
    lm_datasets = tokenized_datasets.map(group_texts, batched=True)
    print(lm_datasets)

    # Training Arguments + Trainer
    batch_size = 64
    logging_steps = len(lm_datasets["train"]) // batch_size
    training_args = TrainingArguments(
        output_dir="./models/mlm-fine-tuned-roberta",
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        fp16=True,
        logging_steps=logging_steps,
    )
    trainer = Trainer(
        model=MODEL,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["test"],
        data_collator=DATA_COLLATOR,
        tokenizer=TOKENIZER
    )
    trainer.train()
    trainer.evaluate()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', "--input", required=False,
                    help=".csv file with data for fine-tuning")
    ap.add_argument('-f', "--folder", required=False,
                    help="folder path output for file saving")
    # ap.add_argument('-m', "--model", required=False,
    #                 help="pre-trained model name")
    args_main = vars(ap.parse_args())

    start_time = datetime.now()
    main(input=args_main["input"], folder=args_main["folder"])
    end_time = datetime.now()
    print(f"Process started at {start_time}, ended at {end_time}, took {end_time-start_time}")
    

