import json
from copy import deepcopy
import os


example_config = {
    # logreg
    "C": 2.0,
    "max_iter": 100,

    # vectorizer
    "max_features": 100,
    "ngram_range": (1, 3),
    "min_df": 1,
    "max_df": 1.0,

    # train setup
    "test_size": 0.33,

    "use_old_train": False,
    "create_val": True
}

k = 129
for C in [1.0, 2.0, 10.0]:
    for max_iter in [1000, 10_000]:
        for max_feats in [10000, 20000]:
            for ngram_range in [(1, 5), (1, 7)]:
                for min_df in [2, 4, 6]:
                    new_conf = deepcopy(example_config)
                    new_conf["C"] = C
                    new_conf["max_iter"] = max_iter
                    new_conf["max_features"] = max_feats
                    new_conf["ngram_range"] = ngram_range
                    new_conf["min_df"] = min_df

                    k += 1
                    os.makedirs(f'./configs3/', exist_ok=True)
                    with open(f'./configs3/config_{k}.json', 'w') as f:
                        json.dump(new_conf, f)
                    with open(f'./run_configs3.sh', '+a') as f:
                        f.write(f'OMP_NUM_THREADS=1 PYTHONPATH=./ python train_tfidf.py --config ./configs3/config_{k}.json\n')

print(f'Generated {k} configs')
