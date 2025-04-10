import os
import json
import random
from job_scp import *

json_save_path = 'json'
raw_filename = 'et_instruct_164k_txt.json'
json_save_all = 'et_164k_txt_1w.json'
all_file = []

clip_list = {}
clip_count = {
    1: 1000,
    2: 1000,
    3: 1300,
    4: 2500,
    5: 1500,
    6: 1200,
    7: 800,
    8: 550,
    9: 400,
    10: 300,
}

with open(raw_filename, 'r') as f:
    all_rec = json.load(f)

clip_list.update(
    {
        1:
            [
                rec
                for rec in all_rec
                if ((rec['task'] == 'tvg' and rec['source'] != 'queryd' and rec['source'] != 'naq') or rec['task'] == 'tal') and 'tgt' in rec and len(rec['tgt']) // 2 == 1
            ]
    }
)

for num in range(2, 11):
    clip_list.update(
        {
            num:
                [
                    rec
                    for rec in all_rec
                    if rec['task'] == 'tal' and 'tgt' in rec and len(rec['tgt']) // 2 == num
                ]
        }
    )

# for key in clip_list:
#     [random.shuffle(clip_list[key]) for _ in range(10)]
#     clip_list[key] = selcet_ratio(clip_list[key])
#     clip_list[key] = clip_list[key][:clip_count[key]]

for key in clip_list:
    [random.shuffle(clip_list[key]) for _ in range(10)]
    clip_list[key] = selcet_ratio(clip_list[key])

# all_file.extend(clip_list[1])
# all_file.extend(clip_list[2])

for index in range(1, 3):
    clip_list[index] = [job_0(clip) for clip in clip_list[index]]

for index in range(3, 11):
    processed_records = []
    for rec in clip_list[index]:
        selected_job = random.choices(job_list, weights=job_probabilities, k=1)[0]
        processed_rec = selected_job(rec)
        if processed_rec:
            processed_records.append(processed_rec)
    clip_list[index] = processed_records
    # all_file.extend(processed_records)


for key in clip_list:
    [random.shuffle(clip_list[key]) for _ in range(10)]
    clip_list[key] = selcet_ratio(clip_list[key])
    # print(f"pre {key}: {len(clip_list[key])}")
    clip_list[key] = clip_list[key][:clip_count[key]]
    print(f"|{key}| {len(clip_list[key])} |")
    all_file.extend(clip_list[key])

with open(json_save_all, 'w') as f:
    print(f'|all|{len(all_file)}|')
    [random.shuffle(all_file) for _ in range(10)]
    json.dump(all_file, f, indent=4)