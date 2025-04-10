import os
import json
import tqdm
from typing import Iterable

org_file_list = ["etbench_txt_v1.0.json", ] #"et_instruct_164k_vid.json"
src_file = "file_json/test_2.json"
tar_file = "file_json/test_2_org.json"

json_data_org = []
json_data_tar = []

def flatten(nested):
    flattened = []
    for item in nested:
        if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
            flattened.extend(flatten(item))
        else:
            flattened.append(item)
    return flattened

for json_file in org_file_list:
    with open(json_file, 'r') as f:
        json_data = json.load(f)
        json_data_org.extend(json_data)

with open(src_file, 'r') as f:
    json_data_src = json.load(f)

with tqdm.tqdm(total=len(json_data_src), desc="Processing Videos", unit="video", dynamic_ncols=True) as pbar:
    for src_data in json_data_src:
        temp_data = src_data
        for item in json_data_org:
            if item["video"] == src_data["video"]:
                src_data["q"] = item["q"]
                src_data["conversations"][0]["value"] = item["q"]
                # item['tgt'] = flatten(item['tgt'])
                json_data_tar.append(src_data)
                break
        pbar.update(1)

with open(tar_file, 'w') as f:
    json.dump(json_data_tar, f, indent=4)