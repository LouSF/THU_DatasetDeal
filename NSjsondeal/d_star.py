import os
import re
import sys
import random
import json
import multiprocess
import tqdm
from pandas import options

json_path = "/Users/lsf/PycharmProjects/DatasetJson/NSjsondeal/Complise/star"
save_path = "/Users/lsf/PycharmProjects/DatasetJson/NSjsondeal/middle/star"
json_file_list = os.listdir(json_path)
json_file_list = [rec for rec in json_file_list if rec.endswith(".json") and not rec.endswith("fix.json")]
json_label = json_path.split('/')[-1]

select_type = ["Sequence", "Prediction", "Interaction",]

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
                if rec['question_id'].split('_')[0] in select_type
            ]

    # change dataset
    fixed_json_files_dict = {}
    for index, json_list in json_files_dict.items():
        fixed_json_list = []
        for rec in tqdm.tqdm(json_list, total=len(json_list), desc=f"Processing {index}"):

            fixed_options = [item["choice"] for item in rec['choices']]
            fixed_options = list(set(fixed_options))
            random.shuffle(fixed_options)

            fixed_json_list.append(
                {
                    "dataset": json_label,
                    "task": rec['question_id'].split('_')[0],
                    'id': rec['question_id'],
                    "video": rec['video_id'] + '.mp4',
                    'tgt': [
                        rec['start'],
                        rec['end'],
                    ],
                    "original_question": rec['question'],
                    "original_answer": rec['answer'],
                    "conversations": [
                        {
                            "from": "human",
                            "value": "".join([f"Focus on the segment {rec['start']} - {rec['end']}.", " ", rec['question'],]),
                        },
                        {
                            "from": "gpt",
                            "type": "select_option",
                            "value": fixed_options.index(rec['answer']),
                        },
                    ],
                    "options": fixed_options,
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