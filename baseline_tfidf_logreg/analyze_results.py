from pathlib import Path
import json 
import pandas as pd
from collections import defaultdict

df = defaultdict(list)
for result_file in Path('results').iterdir():
    with open(result_file) as f:
        dat = json.load(f)
    for k, v in dat.items():
        df[k].append(v)
    df["name"].append(result_file.name)

df = pd.DataFrame(df)
print(df.sort_values(by="val_acc", ascending=False).head())
print(df.sort_values(by="test_acc", ascending=False).head())
print(df.sort_values(by="test_acc2", ascending=False).head())

