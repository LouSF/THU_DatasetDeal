import os

from sympy import false

import json
import numpy as np

json_path = "et_164k_txt_1w_raw_unique.json"

with open(json_path, 'r') as json_file:
    json_data = json.load(json_file)


acc = 0
una = 0

video_acc = 0
video_una = 0

for record in json_data:
    F = False
    for index in range(0, len(record['tgt']), 2):
        si = False
        for t in np.linspace(0.0, record['duration'], 32):
            if (record['tgt'][index] <= t) and (record['tgt'][index + 1] >= t):
                si = True
        if si:
            acc += 1
        else:
            una += 1

    if F:
        video_acc += 1
    else:
        video_una += 1

print("-------------Target-------------")
print(acc)
print("-------------Missin-------------")
print(una)
print("-------------RadioS-------------")
print(acc/(acc + una))
print("--------------------------------")

print("-------------Target-------------")
print(video_acc)
print("-------------Missin-------------")
print(video_una)
print("-------------RadioS-------------")
print(video_acc/(video_acc + video_una))
print("--------------------------------")