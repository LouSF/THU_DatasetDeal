import os
import json
import random
from job_scp import calculate_time_segments_ratio

json_save_all = 'et_164k_txt_1w_raw.json'
raw_filename = 'et_instruct_164k_txt.json'
dealed_data = []
clip_list = {}
clip_count = {
    1: 650,
    2: 1000,
    3: 1000,
    4: 2500,
    5: 1600,
    6: 1200,
    7: 800,
    8: 550,
    9: 400,
    10: 300,
}


def selcet_ratio(input_rec: list, method = 'raw') -> list:
    result = []
    for rec in input_rec:
        ratio = calculate_time_segments_ratio(rec['duration'], rec['tgt'])
        if ratio <= 16:
            rec.update(
                {
                    'ratio': ratio
                }
            )
            result.append(rec)
    return result

with open(raw_filename, "r") as infile:
    all_data = json.load(infile)

clip_list.update(
    {
        1:
            [
                rec
                for rec in all_data
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
                    for rec in all_data
                    if rec['task'] == 'tal' and 'tgt' in rec and len(rec['tgt']) // 2 == num
                ]
        }
    )

for key in clip_list:
    [random.shuffle(clip_list[key]) for _ in range(10)]
    clip_list[key] = selcet_ratio(clip_list[key])
    clip_list[key] = clip_list[key][:clip_count[key]]

count = 0
for rec in clip_list:
    count += len(clip_list[rec])
    dealed_data.extend(clip_list[rec])
    print(f"| {rec} | {len(clip_list[rec])} |")
print(f"| all | {count} |")

with open(json_save_all, 'w') as f:
    [random.shuffle(clip_list[key]) for _ in range(10)]
    json.dump(dealed_data, f, indent=4)