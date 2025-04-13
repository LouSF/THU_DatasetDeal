import os
import sys
import random
import json
import multiprocess
import re
import tqdm
from openai import OpenAI
import time


# DEEPSEEK_API_KEY = "sk-yvwmnxbbkacqcjpdeqzxssboeicgcjpumlomplhaatyenfme"
# BASE_URL = "https://api.siliconflow.cn/v1"

DEEPSEEK_API_KEY = "sk-a2d903cd84e7430fb4b661dfde17e031"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

MODEL = "qwen-max"
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)


dataset_file = []

json_path = "/Users/lsf/PycharmProjects/DatasetJson/NSjsondeal/Complise/nextqa"
json_file_list = os.listdir(json_path)
json_file_list = [rec for rec in json_file_list if rec.endswith(".json") and not rec.startswith("fixed")]
json_label = json_path.split('/')[-1]

select_type = ["TN", "TP", "TC", "CW",]

import json
from typing import Dict, Any


def validate_action_json(output_json: Dict[str, Any]) -> bool:
    try:
        if "actions" not in output_json:
            return False

        for action in output_json["actions"]:
            required_fields = ["subject", "verb", "object", "purpose"]
            if not all(field in action for field in required_fields):
                return False

        return True
    except Exception:
        return False

def generate_with_deepseek(
        prompt: str,
        model: str = "qwen-max",
        max_retries: int = 3,
) -> Dict[str, Any]:
    system_prompt = """
    You MUST extract action components from the user's sentence and return STRICT JSON format.
    Follow this EXACT structure:

    {
        "actions": [{
            "subject": "the person/object doing the action",
            "verb": "the action verb",
            "object": "the target of the action",
            "purpose": "the purpose (if any)"
        }]
    }

    EXAMPLE INPUT: 
    "The girl approaches Santa Claus to take a photo."

    EXAMPLE OUTPUT:
    {
        "actions": [{
            "subject": "the girl",
            "verb": "approaches",
            "object": "Santa Claus",
            "purpose": "to take a photo"
        }]
    }

    IMPORTANT RULES:
    1. ALWAYS include ALL fields (subject, verb, object, purpose).
    2. If purpose is missing, use "".
    3. NEVER add extra fields.
    """

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
            )

            result = json.loads(response.choices[0].message.content)

            if validate_action_json(result):
                print(f"Attempt {attempt + 1}: Success!")
                return result
            else:
                print(f"Attempt {attempt + 1}: Invalid format, retrying...")

        except Exception as e:
            print(f"Attempt {attempt + 1}: Error - {e}")

    raise ValueError(f"Failed after {max_retries} retries.")


def generate_question_templates(original_question, options, answer, function):
    templates_func1 = [
        "Describe two consecutive events in the video where the first action directly triggers the second.",
        "One thing happens in the video, and another thing happens after it. What are these two things?",
    ]

    templates_func2 = [
        "Complete the sequence: The video first shows ______, then later shows ______.",
        "The main event is ______. What happens immediately after this?",
        "Complete the sequence: The video first shows ______, then later shows ______.",
    ]

    templates_question, templates_options, templates_answer = [], [], []

    # func1 视频中有一件事情发生，在此事以后有另一件事情发生，这两件事情分别是什么？ _ 选项 A B C D E
    if function == 'func1':
        pass

    # func2 填空：在 _ 后发生了 _ 选项 A B C D E
    if function == 'func2':
        pass



    return templates_question, templates_options, templates_answer


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
            if rec['metadata']['type'] in select_type
        ]


    # change dataset
    fixed_json_files_dict = {}
    for index, json_list in json_files_dict.items():
        fixed_json_list = []
        for rec in tqdm.tqdm(json_list, total=len(json_list), desc=index):
            try:
                question, option, answer = generate_question_templates(rec['question'], rec['answer'], rec['options'], "func1")
                fixed_json_list.append(
                    {
                        "dataset": json_label,
                        "task": rec['metadata']['type'],
                        'id': rec['video'],
                        "video": rec['video'],
                        "question": question,
                        "answer": answer,
                        "options": rec['options'],
                        "conversations": [
                            {
                                "from": "human",
                                "value": rec['question'],
                            },
                            {
                                "from": "gpt",
                                "value": rec['options'][rec['answer']],
                            },
                        ],
                        "conv2": generate_with_deepseek(rec['question']),
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
        with open(os.path.join(json_path, f'fixed_{index}_gpt_fixed'), 'w') as file:
            json.dump(json_list, file, indent=4)

    for index, json_list in fixed_json_files_dict.items():
        print(f"|{index}|{len(json_list)}|")
        select_type_num = {key: 0 for key in select_type}
        for rec in json_list:
            select_type_num[rec["task"]] += 1
        for index, json_list in select_type_num.items():
            print(f"|^{index}^|{json_list}|")




if __name__ == "__main__":
    main()
