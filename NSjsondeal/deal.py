import os
import sys
import random
import json
import multiprocess

json_path = "/Users/lsf/PycharmProjects/DatasetJson/NSjsondeal/Complise/star"
json_file_list = os.listdir(json_path)
json_file_list = [rec for rec in json_file_list if rec.endswith(".json") and not rec.startswith("fixed")]
json_label = json_path.split('/')[-1]


prompt_mop = {
    "Sequence": {
        "T1": r"^Which object did the person (\w+) after they (\w+)ed the (\w+)?",
        "T2": r"^Which object did the person (\w+) before they (\w+)ed the (\w+)?",
        "T3": r"^What happened after the person (\w+)ed the (\w+)?",
        "T4": r"^What happened before the person (\w+)ed the (\w+)?",
        "T5": r"^What did the person do to the (\w+) after (\w+)ing the (\w+)?",
        "T6": r"^What did the person do to the (\w+) before (\w+)ing the (\w+)?",
    },
    "Prediction": {
        "T1": r"^What will the person do next?",
        "T2": r"^What will the person do next with the (\w+)?",
        "T3": r"^Which object would the person (\w+) next?",
        "T4": r"^Which object would the person (\w+) next after they (\w+) the (\w+)?",
    }
}

templates_func1 = [
        "Describe two consecutive events in the video where the first action directly triggers the second.",
        "One thing happens in the video, and another thing happens after it. What are these two things?",
    ]

templates_func2 = [
        "Complete the sequence: The video first shows ______, then later shows ______.",
        "The main event is ______. What happens immediately after this?",
        "Complete the sequence: The video first shows ______, then later shows ______.",
    ]

select_type = ["Sequence", "Prediction", ]


def main():

    # read files
    json_files_dict = {}
    for json_file_name in json_file_list:
        with open(os.path.join(json_path, json_file_name), 'r') as file:
            json_files_dict[json_file_name] = json.load(file)

    # split useful data
    for index, json_list in json_files_dict.items():
        json_files_dict[index] = [
            rec for rec in json_list
            if rec['question_id'].startswith('Sequence') or
               rec['question_id'].startswith('Prediction')
            ]

    # change dataset
    fixed_json_files_dict = {}
    for index, json_list in json_files_dict.items():
        fixed_json_list = []
        for rec in json_list:
            try:

                dataset_type = rec['question_id'].split('_')[0]
                task_id = rec['question_id'].split('_')[1]



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
                        "conversations": [
                            {
                                "from": "human",
                                "value": rec['question'],
                            },
                            {
                                "from": "gpt",
                                "value": rec['answer'],
                            }
                        ],
                    }
                )
            except Exception as e:
                print(e)

        fixed_json_files_dict.update(
            {
                index: fixed_json_list
            }
        )

    for index, json_list in fixed_json_files_dict.items():
        print(f"|{index}|{len(json_list)}|")
        with open(os.path.join(json_path, f'fixed_{index}'), 'w') as file:
            json.dump(json_list, file, indent=4)




if __name__ == "__main__":
    main()