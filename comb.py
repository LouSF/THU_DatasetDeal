import os
import json
import random

path = 'new_comb'
list_file = os.listdir('new_comb')
list_file = [os.path.join(path, i) for i in list_file if i.endswith('.json')]
print(list_file)

all_file = []

for i in list_file:
    with open(i, 'r') as f:
        all_file.extend(json.load(f))

[random.shuffle(all_file) for _ in range(100)]

for index, rec in enumerate(all_file):
    rec.update({'id': index})

with open('et_instruct_164k_txt_329.json', 'w') as f:
    json.dump(all_file, f, indent=4)