from pathlib import Path
import json 
import pandas as pd
from collections import defaultdict
import numpy as np


df = defaultdict(list)
for result_file in Path('results').iterdir():
    with open(result_file) as f:
        dat = json.load(f)
    
    for ind, el in enumerate(dat):
        df[ind].append(max([np.round(a, 3 )for a in el]))
    df["name"].append(result_file.name)

df = pd.DataFrame(df)
print(df)
# print(df.sort_values(by="val_acc", ascending=False).head())
# print(df.sort_values(by="test_acc", ascending=False).head())
# print(df.sort_values(by="test_acc2", ascending=False).head())

