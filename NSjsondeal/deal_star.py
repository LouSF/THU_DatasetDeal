import os
import re
import sys
import random
import json
import nltk
import multiprocess
import tqdm

json_path = "/Users/lsf/PycharmProjects/DatasetJson/NSjsondeal/Complise/star"
save_path = "/Users/lsf/PycharmProjects/DatasetJson/NSjsondeal/Complise/star/save"
json_file_list = os.listdir(json_path)
json_file_list = [rec for rec in json_file_list if rec.endswith(".json") and not rec.endswith("fix.json")]
json_label = json_path.split('/')[-1]

select_type = ["Sequence", "Prediction", ]

prompt_mop = {
    "Sequence": {
        "T1": r"^Which object did the person (.*?) after they (.*?) the (.*?)\?",
        "T2": r"^Which object did the person (.*?) before they (.*?) the (.*?)\?",
        "T3": r"^What happened after the person (.*?) the (.*?)\?",
        "T4": r"^What happened before the person (.*?) the (.*?)\?",
        "T5": r"^What did the person do to the (.*?) after (.*?) the (.*?)\?",
        "T6": r"^What did the person do to the (.*?) before (.*?) the (.*?)\?",
    },
    "Prediction": {
        "T1": r"^What will the person do next?",
        "T2": r"^What will the person do next with the (.*?)\?",
        "T3": r"^Which object would the person (.*?) next\?",
        "T4": r"^Which object would the person (.*?) next after they (.*?) the (.*?)\?",
    }
}

prompt_target = {
    # t time
    # v v
    # d ved
    # i ving
    # n n
    "Sequence":[
        ("During {} to {}, the person did two things. Please list them sequentially.", 'tt',),
        ("What did the person do from {} to {}? List the things they do sequentially.", 'tt',),
        ("The person did A before B between {} and {}. What are A and B??", 'tt',),
        ("The person did A after B between {} and {}. What are A and B?", 'tt',),
        ("Focus on the segment {} - {}. What did the person do after they {}?", 'ttd',),
        ("Focus on the segment {} - {}. What did the person do before they {}?", 'ttd',),
        ("Answer the above question according to the video. Only use words from the following words to organize your answer.", '',),
    ],
    "Prediction":[
        ("Which object did the person {} after they {} the ___ ? The ___.", 'vd',),
        ("Which object did the person {} before they {} the ___ ? The ___.", 'vd'),
        ("The person ___ the ___ after they ___ the ___.", '',),
        ("The person ___ the ___ before they ___ the ___.", '',),
        ("Choose words from the following words to fill in the blanks according to segment {} - {} of the video.", 'tt',),
    ]
}



def generate_question_templates(id, original_question, options, answer):
    target_id = id.split('_')
    target_template = prompt_mop[target_id[0]][target_id[1]]

    match = re.search(target_template, original_question)
    if not match:
        raise ValueError(
            f"Template '{target_template}' not found in question: '{original_question}'"
        )

    fixed_question = original_question
    last_pos = 0
    parts = []

    spans = [match.span(i) for i in range(1, match.lastindex + 1)] if match.lastindex else []

    for start, end in spans:
        parts.append(fixed_question[last_pos:start])
        parts.append("____")
        last_pos = end

    parts.append(fixed_question[last_pos:])
    fixed_question = "".join(parts)

    answer_fixed = []
    matched_groups = match.groups() if match.lastindex else ()
    answer_fixed.extend(matched_groups)
    answer_fixed.append(answer.lower())
    answer = dict()
    answer.update(
        {
            "respond": answer_fixed
        }
    )

    options_list = [item["choice"] for item in options]
    options_list.extend(matched_groups)
    options_list = list(set(options_list))
    options_list = [rec.lower() for rec in options_list]
    random.shuffle(options_list)

    options_select = [options_list.index(i) for i in answer_fixed if i in options_list]

    answer.update(
        {
            "select": options_select
        }
    )

    return fixed_question, options_list, answer


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
        for rec in tqdm.tqdm(json_list, total=len(json_list), desc=f"Processing {index}"):
            try:

                fixed_question, fixed_options, fixed_answers = generate_question_templates(rec['question_id'], rec['question'], rec['choices'], rec['answer'])

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
                                "value": fixed_question,
                            },
                            {
                                "from": "gpt",
                                "type": "select_option",
                                "value": fixed_answers["select"],
                            },
                            {
                                "from": "gpt",
                                "type": "text",
                                "value": fixed_answers["respond"],
                            }
                        ],
                        "options": fixed_options,
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
        with open(os.path.join(save_path, f'fixed_{index}'), 'w') as file:
            json.dump(json_list, file, indent=4)




if __name__ == "__main__":
    main()