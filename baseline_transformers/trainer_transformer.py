import os

import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from dataset import *
from model import *
from trainer import Trainer

import argparse
import json 
import numpy as np


torch.manual_seed(42)

# ai-forever/sbert_large_mt_nlu_ru

example_config = {
    "model_name": "ai-forever/sbert_large_mt_nlu_ru",
    "batch_size": 16,
    "max_len": 128, 
    "model_config": {
        "num_classes": 5,
        "dropout_rate": 0.5,
        "feat_dim": 1024
    }, 
    "trainer_config": {
        "lr": 1e-05,
        "n_epochs": 9,
        "weight_decay": 0.01,
        "device": "cuda",
        "seed": 42, 
        "weights": [6, 10, 4, 3, 1], 
        "label_smoothing": 0.05, 
        "num_steps": 100
    },
    "train_path": None,
    "test_path": None,
    "create_val": True,
}


def load_config(config_path):
    # load json config file
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


PATH = "../data/imdb_csv"

def main(config_path):
    config = load_config(config_path)
    # config["trainer_config"]["n_epochs"] = 3

    model_config = config["model_config"]
    trainer_config = config["trainer_config"]
    MAX_LEN = config["max_len"]
    BATCH_SIZE = config["batch_size"]
    MODEL_NAME = config["model_name"]

    train_path = config.get("train_path", None)
    test_path = config.get("test_path", None)
    create_val = config.get("create_val", True)

    # Split train, val tf-idf
    test_data = pd.read_csv(os.path.join(PATH, "test.csv"))
    if train_path is None:
        train_data = pd.read_csv(os.path.join(PATH, "train.csv"))
    else:
        train_data = pd.read_csv(train_path)

    if test_path is not None:
        test_data2 = pd.read_csv(test_path)
                

    train_split, val_split = train_test_split(train_data, test_size=0.33, random_state=42)
    if not create_val:
        train_split = train_data

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, truncation=True, do_lower_case=True)

    # Creating datasets and dataloaders

    train_dataset = FiveDataset(train_split, tokenizer, MAX_LEN)
    val_dataset = FiveDataset(val_split, tokenizer, MAX_LEN)
    test_dataset = FiveDataset(test_data, tokenizer, MAX_LEN)
    if test_path is not None:
        test_dataset2 = FiveDataset(test_data2, tokenizer, MAX_LEN)

    train_params = {"batch_size": BATCH_SIZE,
                    "shuffle": True,
                    "num_workers": 0
                    }

    test_params = {"batch_size": BATCH_SIZE,
                "shuffle": False,
                "num_workers": 0
                }

    train_dataloader = DataLoader(train_dataset, **train_params)
    val_dataloader = DataLoader(val_dataset, **test_params)
    test_dataloader = DataLoader(test_dataset, **test_params)
    if test_path is not None:
        test_dataloader2 = DataLoader(test_dataset2, **test_params)

    # Loading pretrained model from Huggingface
    model = ModelForClassification(
        MODEL_NAME,
        config=model_config
    )

    trainer_config["batch_size"] = BATCH_SIZE
    # Creating Trainer object and fitting the model
    t = Trainer(trainer_config)

    val_dataloaders = {
        "val": val_dataloader,
        "test": test_dataloader
    }
    if test_path is not None:
        val_dataloaders["test2"] = test_dataloader2
                
    t.fit(
        model,
        train_dataloader,
        val_dataloaders
    )

    # val_metrics = t.val_epoch(val_dataloader)
    # test_metrics = t.val_epoch(test_dataloader)

    print(t.history["val_acc"], t.history["test_acc"], t.history["test2_acc"] if test_path is not None else "No test 2")

    os.makedirs("results", exist_ok=True)
    from pathlib import Path
    with open(f"results/{Path(config_path).name}", "w") as f:
        json.dump([t.history["val_acc"], t.history["test_acc"], t.history["test2_acc"] if test_path is not None else "No test 2"], f)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", type=str, default="config.json")
    args = arg_parser.parse_args()
    config_path = args.config
    main(config_path)
          