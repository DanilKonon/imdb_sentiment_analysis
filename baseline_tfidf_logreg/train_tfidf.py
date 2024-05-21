 
# # Import Libraries
 
import os
import pandas as pd
import numpy as np
import nltk
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import json
from pathlib import Path


# nltk.download("stopwords")
path = "../data/imdb_csv"


def load_config(config_path):
    # load json config file
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def compute_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def main(config_path):
    config = load_config(config_path)

    create_val = config.get("create_val", True)
    use_old_train = config.get("use_old_train", True)

    # Read Data 
    train = pd.read_csv(os.path.join(path, "train.csv"))
    test = pd.read_csv(os.path.join(path, "test.csv"))
    test2 = pd.read_csv(os.path.join(path, "imdb_new_reviews_v2.csv"))

    if not use_old_train:
        train, test2 = test2, train
    
    print(f"train len: {len(train)}")

    # # Text Preprocessing
    stopwords = nltk.corpus.stopwords.words("english")

    # Init tf-idf
    vect_word = TfidfVectorizer(
        max_features=config["max_features"],
        lowercase=True,
        analyzer="word",
        stop_words=stopwords,
        ngram_range=config["ngram_range"],
        dtype=np.float32,
        max_df=config["max_df"],
        min_df=config["min_df"],
    )

    # Split train, val tf-idf
    X_train_texts, X_val_texts, y_train, y_val = train_test_split(train["text"], train["label"], test_size=config["test_size"], random_state=42)
    if not create_val:
        X_train_texts, y_train = train["text"], train["label"]

    print(f"train len: {len(X_train_texts)}")


    X_train = vect_word.fit_transform(X_train_texts)
    X_val = vect_word.transform(X_val_texts)

    # Map tf-idf on test
    X_test = vect_word.transform(test["text"])
    y_test = test["label"]

    X_test2 = vect_word.transform(test2["text"])
    y_test2 = test2["label"]

    # Init logreg model
    logreg = LogisticRegression(
        C=config["C"],
        random_state=42,
        max_iter=config["max_iter"],
    )

    # Train Model
    logreg.fit(X_train, y_train)

    # Predict probabilities
    train_acc = compute_accuracy(y_train, logreg.predict(X_train))
    val_acc = compute_accuracy(y_val, logreg.predict(X_val))
    test_acc = compute_accuracy(y_test, logreg.predict(X_test))
    test_acc2 = compute_accuracy(y_test2, logreg.predict(X_test2))
    
    return train_acc, val_acc, test_acc, test_acc2

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default="config.json")
    args = parser.parse_args()
    config = args.config
    train_acc, val_acc, test_acc, test_acc2 = main(config)
    print("Train/Validation/Test Accuracy:", train_acc, val_acc, test_acc, test_acc2)
    Path("results").mkdir(parents=True, exist_ok=True)
    with open(f"results/{config.name}", "w") as f:
        json.dump({"train_acc": train_acc, "val_acc": val_acc, "test_acc": test_acc, "test_acc2": test_acc2}, f)
