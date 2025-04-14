import os
import re
import sys
import random
import json
import multiprocess
import tqdm
from pandas import options

json_path = "/Users/lsf/PycharmProjects/DatasetJson/NSjsondeal/Complise/nextqa"
save_path = "/Users/lsf/PycharmProjects/DatasetJson/NSjsondeal/middle/nextqa"
json_file_list = os.listdir(json_path)
json_file_list = [rec for rec in json_file_list if rec.endswith(".json") and not rec.endswith("fix.json")]
json_label = json_path.split('/')[-1]

select_type = ["TP", "TN", "TC", ]

def main():

    # read files
    json_files_dict = {}
    for json_file_name in json_file_list:
        with open(os.path.join(json_path, json_file_name), 'r') as file:
            json_files_dict[json_file_name] = json.load(file)

    # split useful data
    for index, json_list in json_files_dict.items():
        if 'train' in index:
            json_files_dict[index] = [
                rec for rec in json_list
                if rec['metadata']['type'] in select_type
            ]

    # change dataset
    fixed_json_files_dict = {}
    for index, json_list in json_files_dict.items():
        fixed_json_list = []
        for rec in tqdm.tqdm(json_list, total=len(json_list), desc=f"Processing {index}"):

            fixed_json_list.append(
                {
                    "dataset": json_label,
                    "task": rec['metadata']['type'],
                    'id': rec["video"],
                    "video": rec["video"],
                    'tgt': None,
                    "original_question": rec['question'],
                    "original_answer": rec['answer'],
                    "conversations": [
                        {
                            "from": "human",
                            "value": rec['question'],
                        },
                        {
                            "from": "gpt",
                            "type": "select_option",
                            "value": rec['answer'],
                        },
                    ],
                    "options": rec["options"],
                }
            )
            # except Exception as e:
            #     print(e)

            fixed_json_files_dict.update(
                {
                    index: fixed_json_list
                }
            )

        for index, json_list in fixed_json_files_dict.items():
            print(f"|{index}|{len(json_list)}|")
            with open(os.path.join(save_path, f'fixed_{index}'), 'w') as file:
                json.dump(json_list, file, indent=4)

if __name__ == "__main__":
    main()