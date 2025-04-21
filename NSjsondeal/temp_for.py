import os
import json
path = '/Users/lsf/PycharmProjects/DatasetJson/NSjsondeal/Complise/star/save/fixed_train.json'
sath = '/Users/lsf/PycharmProjects/DatasetJson/NSjsondeal/Complise/star/save/fixed_train_SCT.json'

flist = {}
with open(path, 'r') as f:
    all_data = json.load(f)

for rec in all_data:
    target = rec["fixed_type"]
    if target not in flist:
        flist.update(
            {
                target: [],
            }
        )
    rec.update(
        {
            "check_target": target,
        }
    )
    flist[target].append(rec)

selc_data = []

for key in flist:
    print(key)
    selc_data.extend(flist[key][:1])

with open(sath,'w') as f:
    json.dump(selc_data, f, indent=4)