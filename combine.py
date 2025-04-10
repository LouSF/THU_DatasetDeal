import os
import json
import random

json_path = "/Users/lsf/Downloads/data"

train_list = [
    "et_tal_activitynet.json",
    "et_tvg_didemo.json",
    "et_tal_hacs.json",
    "et_tvg_tacos.json",
]

val_list = [
    "val/et_val_qvhighlights.json",
    "val/et_val_thumos.json",
]

train_json = {}
val_json = {}
train = []
val = []
val_val = []

for item in train_list:
    with open(os.path.join(json_path, item), "r") as file:
        json_data = json.load(file)
        print(item)
        print(len(json_data))
        train_json[item] = json_data


for item in train_json:
    split_idx = int(len(train_json[item]) * 0.8)
    trainT = train_json[item][:split_idx]
    valT = train_json[item][split_idx:]
    train.extend(trainT)
    val.extend(valT)
    print(item)
    print(f"train:{len(train)}")
    print(f"test:{len(val)}")

for item in val_list:
    with open(os.path.join(json_path, item), "r") as file:
        json_data = json.load(file)
        # print(item)
        # print(len(json_data))
        val_json[item] = json_data


for item in val_json:
    val_data = val_json[item]
    val_val.extend(val_data)
    print(item)
    print(f"item:{len(val_json[item])}")

train += val_val

print(f"all train:{len(train)}")
print(f"all test:{len(val)}")

random.shuffle(train)
random.shuffle(val)

with open(os.path.join(json_path, "train.json"), "w") as file:
    json.dump(train_json, file, indent=4)

with open(os.path.join(json_path, "test.json"), "w") as file:
    json.dump(val, file, indent=4)