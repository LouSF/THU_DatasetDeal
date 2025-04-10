import json
import random

with open("et_instruct_164k_vid.json", "r") as file:
    json_data = json.load(file)

random.shuffle(json_data)


json_tal = []
json_tvg = []
json_all = []
tal_sum = 0
tvg_sum = 0


for data in json_data:
    if data["task"] == "tal":
        json_tal.append(data)
        json_all.append(data)
        tal_sum += 1
    if data["task"] == "tvg" and (data["source"] == "didemo" or data["source"] == "tacos"):
        json_tvg.append(data)
        json_all.append(data)
        tvg_sum += 1

# with open("et_instruct_164k_vid_tal.json", "w") as file_tal:
#     # file_tal.write(json.dumps(json_tal))
#     json.dump(json_tal, file_tal, indent=4)
# with open("et_instruct_164k_vid_tvg.json", "w") as file_tvg:
#     # file_tvg.write(json.dumps(json_tvg))
#     json.dump(json_tvg, file_tvg, indent=4)
#

# with open("et_instruct_164k_vid_spilt.json", "w") as file_all:
#     # file_tvg.write(json.dumps(json_all))
#     json.dump(json_all, file_all, indent=4)


print(f"tvg_manual: {tvg_sum} \n"
      f"tal: {tal_sum} \n")
