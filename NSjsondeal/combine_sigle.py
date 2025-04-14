import os
import json
import random

file_list = [
    "nextqa",
    "PercepTest",
    "star",
]

file_train_list = [os.path.join(os.path.join('middle', _), "fixed_train.json") for _ in file_list]
file_val_list = [os.path.join(os.path.join('middle', _), "fixed_val.json") for _ in file_list]

train_rec = []
val_rec = []

for train_f in file_train_list:
    with open(train_f, 'r') as f:
        _ = json.load(f)
        random.shuffle(_)
        train_rec.extend(_[:500])

for val_f in file_val_list:
    with open(val_f, 'r') as f:
        _ = json.load(f)
        random.shuffle(_)
        val_rec.extend(_[:250])


with open(os.path.join(os.path.join('sigle', 'train_comb.json')), 'w') as f:
    random.shuffle(train_rec)
    json.dump(train_rec, f, indent=4)

with open(os.path.join(os.path.join('sigle', 'val_comb.json')), 'w') as f:
    random.shuffle(val_rec)
    json.dump(val_rec, f, indent=4)