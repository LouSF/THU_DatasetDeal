import os
import json
path = '/Users/lsf/PycharmProjects/DatasetJson/NSjsondeal/Complise/star/save/fixed_train.json'
sath = '/Users/lsf/PycharmProjects/DatasetJson/NSjsondeal/Complise/star/save/fixed_train_SCT.json'

flist = {
    'S00': [],
    'S01': [],
    'S02': [],
    'S03': [],
    'S04': [],
    'S10': [],
    'S11': [],
    'S12': [],
    'P0': [],
    'P1': [],
    'P2': [],
    'P3': [],
    'P4': [],
    'I0': [],
}
with open(path, 'r') as f:
    all_data = json.load(f)

for rec in all_data:
    flist[rec["fixed_type"]].append(rec)

selc_data = []

for key in flist:
    selc_data.extend(flist[key][:5])

with open(sath,'w') as f:
    json.dump(selc_data, f, indent=4)