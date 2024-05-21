import json
from copy import deepcopy
import os

conf = {
    "model_name": "intfloat/multilingual-e5-large",
    "batch_size": 16,
    "max_len": 128,
    "model_config": {
        "num_classes": 2,
        "dropout_rate": 0.5, 
        "feat_dim": 1024, 
        "is_average": True
    }, 
    "trainer_config": {
        "lr": 1e-06, 
        "n_epochs": 2,
        "weight_decay": 0.07,
        "device": "cuda", 
        "seed": 42, 
        "weights": None, 
        "label_smoothing": 0.0,
        "num_steps": None
    }
}

conf['trainer_config']['n_epochs'] = 10
conf['trainer_config']['num_steps'] = 15000
# new_conf['max_len'] = 256

k = 15
for batch_size in [16, ]:
    for lr in [5e-07, 1e-7]:
        for weight_decay in [0.01, 0.07]:
            for dropout_rate in [0.2, 0.5 ]:
                max_len = 512
                new_conf = deepcopy(conf)
                new_conf["batch_size"] = batch_size
                new_conf["trainer_config"]["lr"] = lr
                new_conf["trainer_config"]["weight_decay"] = weight_decay
                new_conf["max_len"] = max_len
                new_conf["model_config"]["dropout_rate"] = dropout_rate

                k += 1
                os.makedirs(f'./configs2', exist_ok=True)
                with open(f'./configs2/config_{k}.json', 'w') as f:
                    json.dump(new_conf, f)
                with open(f'./run_configs4.sh', '+a') as f:
                    f.write(f'PYTHONPATH=./ OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python trainer_transformer.py --config ./configs2/config_{k}.json\n')

