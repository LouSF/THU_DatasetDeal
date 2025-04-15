import os
import json
import random

from mpmath.functions.zetazeros import count_to

train_file_path = 'sigle/train_comb.json'
val_file_path = 'sigle/val_comb.json'

with open(train_file_path, 'r') as f:
    train_json = json.load(f)

with open(val_file_path, 'r') as f:
    val_json = json.load(f)

count_train = {}
count_val = {}

for _ in train_json:
    nstr = _["dataset"] + "_" + _["task"]
    if nstr not in count_train:
        count_train.update(
            {
            nstr: 0
            }
        )
    count_train[nstr] += 1

for _ in val_json:
    nstr = _["dataset"] + "_" + _["task"]
    if nstr not in count_val:
        count_val.update(
            {
            nstr: 0
            }
        )
    count_val[nstr] += 1

print("| index | count |")
print("| - | - |")
all_c = 0
count_train = {k: count_train[k] for k in sorted(count_train)}
for index, count in count_train.items():
    print(f"| {index} | {count} |")
    all_c += count
print(f"| all | {all_c} |")

print()

print("| index | count |")
print("| - | - |")
all_c = 0
count_val = {k: count_val[k] for k in sorted(count_val)}
for index, count in count_val.items():
    print(f"| {index} | {count} |")
    all_c += count

print(f"| all | {all_c} |")