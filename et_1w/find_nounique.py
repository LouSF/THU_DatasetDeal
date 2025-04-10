import json
from collections import defaultdict

with open("/Users/lsf/PycharmProjects/DatasetJson/et_1w/et_164k_txt_1w_raw_unique.json", "r") as f:
    all_data = json.load(f)

seen = set()
unique_list = []
for d in all_data:
    serialized = json.dumps(d, sort_keys=True)
    if serialized not in seen:
        seen.add(serialized)
        unique_list.append(d)

all_data = unique_list

seen = set()
unique_list = []
for d in all_data:
    serialized = d['video'] + d['conversations'][0]['value']
    if serialized not in seen:
        seen.add(serialized)
        unique_list.append(d)
    else:
        print('X'*10)
        for rec in all_data:
            if serialized == rec['video'] + rec['conversations'][0]['value']:
                # print(json.dumps(rec,indent=4))
                print(rec)
                print('-'*10)
