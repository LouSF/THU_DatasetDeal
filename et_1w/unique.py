import os
import json
import random
from job_scp import calculate_time_segments_ratio

json_all_file = 'et_instruct_164k_txt.json'
json_save_file = 'et_164k_txt_1w_raw_unique.json'
json_output_file = 'et_164k_txt_2w_raw_unique.json'

with open(json_all_file, 'r') as f:
    data_all = json.load(f)

with open(json_save_file, 'r') as f:
    data_base = json.load(f)

seen = set()
unique_list = []
for d in data_base:
    # serialized = json.dumps(d, sort_keys=True)
    serialized = d['video'] + d['conversations'][0]['value']
    if serialized not in seen:
        seen.add(serialized)
        unique_list.append(d)

print(len(unique_list))

clip_list = {}

# clip_list.update(
#     {
#         1:
#             [
#                 rec
#                 for rec in data_all
#                 if ((rec['task'] == 'tvg' and rec['source'] != 'queryd' and rec['source'] != 'naq') or rec['task'] == 'tal') and 'tgt' in rec and len(rec['tgt']) // 2 == 1
#             ]
#     }
# )
clip_list.update(
    {
        1:
            [
                rec
                for rec in data_all
                if (rec['task'] == 'tal') and 'tgt' in rec and len(rec['tgt']) // 2 == 1
            ]
    }
)

for num in range(2, 11):
    clip_list.update(
        {
            num:
                [
                    rec
                    for rec in data_all
                    if rec['task'] == 'tal' and 'tgt' in rec and len(rec['tgt']) // 2 == num
                ]
        }
    )

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

for key in clip_list:
    [random.shuffle(clip_list[key]) for _ in range(10)]
    clip_list[key] = selcet_ratio(clip_list[key])

for index in range(10, 0, -1):
    for i in clip_list[index]:
        # serialized = json.dumps(i, sort_keys=True)
        serialized = i['video'] + i['conversations'][0]['value']
        if serialized not in seen:
            seen.add(serialized)
            unique_list.append(i)
        if len(unique_list) == 20000:
            break
    if len(unique_list) == 20000:
        break

with open(json_output_file, 'w') as f:
    [random.shuffle(unique_list) for _ in range(10)]
    json.dump(unique_list, f, indent = 4)


with open(json_output_file, 'r') as f:
    data_base_change = json.load(f)
    print(len(data_base_change))

clip_list_base = {}
clip_list_base.update(
    {
        1:
            [
                rec
                for rec in data_all
                if (rec['task'] == 'tal') and 'tgt' in rec and len(rec['tgt']) // 2 == 1
            ]
    }
)
for num in range(2, 11):
    clip_list_base.update(
        {
            num:
                [
                    rec
                    for rec in data_all
                    if rec['task'] == 'tal' and 'tgt' in rec and len(rec['tgt']) // 2 == num
                ]
        }
    )
print(f"| Video Clip | 总数 |")
print(f"| - | - |")
for index in clip_list_base:
    print(f"| {index} | {len(clip_list_base[index])} |")