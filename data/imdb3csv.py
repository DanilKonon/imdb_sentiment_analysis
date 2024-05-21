import pandas as pd
from pathlib import Path

path_to_imdb = Path('./aclImdb')


for dataset_type in ["train", "test"]: 
    path_to_dataset = path_to_imdb / dataset_type
    data_for_df = {
        "text": [], "label": []
    }
    for class_name in ["neg", "pos"]:
        path_to_class = path_to_dataset / class_name
        for filename in path_to_class.iterdir():
            with open(filename, 'r') as f:
                review = f.read()
            data_for_df["text"].append(review)
            data_for_df["label"].append(int(class_name == "pos"))
    df = pd.DataFrame(data_for_df)
    Path("./imdb_csv").mkdir(parents=True, exist_ok=True)
    df.to_csv(f"./imdb_csv/{dataset_type}.csv", index=False)

