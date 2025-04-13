import os
import sys
import random
import json
import multiprocessing
import re
from tqdm import tqdm
from openai import OpenAI
import time

# 配置信息
DEEPSEEK_API_KEY = "sk-a2d903cd84e7430fb4b661dfde17e031"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen-max"

# 初始化客户端
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)

# 数据集路径
json_path = "/Users/lsf/PycharmProjects/DatasetJson/NSjsondeal/Complise/nextqa"
select_type = ["TN", "TP", "TC", "CW"]


def validate_action_json(output_json: dict) -> bool:
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


def generate_with_deepseek(prompt: str, max_retries: int = 10) -> dict:
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
                model=MODEL,
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


def process_item(item):
    try:
        question = item['question']
        conv2 = generate_with_deepseek(question)

        return {
            "dataset": "nextqa",
            "task": item['metadata']['type'],
            'id': item['video'],
            "video": item['video'],
            "question": question,
            "answer": item['answer'],
            "options": item['options'],
            "conversations": [
                {
                    "from": "human",
                    "value": item['question'],
                },
                {
                    "from": "gpt",
                    "value": item['options'][item['answer']],
                },
            ],
            "conv2": conv2,
        }
    except Exception as e:
        print(f"Error processing item {item.get('video', 'unknown')}: {e}")
        return None


def main():
    # 读取所有JSON文件
    json_file_list = [f for f in os.listdir(json_path)
                      if f.endswith(".json") and not f.startswith("fixed")]

    # 使用多进程处理
    num_processes = multiprocessing.cpu_count() * 2  # 使用2倍CPU核心数的进程
    print(f"Using {num_processes} processes")

    for json_file_name in json_file_list:
        with open(os.path.join(json_path, json_file_name), 'r') as file:
            original_data = json.load(file)

        # 筛选需要处理的数据
        filtered_data = [rec for rec in original_data
                         if rec['metadata']['type'] in select_type]

        print(f"Processing {json_file_name} with {len(filtered_data)} items")

        # 使用进程池处理数据
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = list(tqdm(pool.imap(process_item, filtered_data),
                                total=len(filtered_data),
                                desc=f"Processing {json_file_name}"))

        # 过滤掉处理失败的项目
        processed_data = [res for res in results if res is not None]

        # 保存结果
        output_file = os.path.join(json_path, f'fixed_{json_file_name}_gpt_fixed_mulit.json')
        with open(output_file, 'w') as file:
            json.dump(processed_data, file, indent=4)

        # 打印统计信息
        print(f"|{json_file_name}|{len(processed_data)}|")
        type_counts = {key: 0 for key in select_type}
        for rec in processed_data:
            type_counts[rec["task"]] += 1
        for task, count in type_counts.items():
            print(f"|^{task}^|{count}|")


if __name__ == "__main__":
    # 在Windows上需要这行代码，在Unix-like系统上不需要
    # multiprocessing.freeze_support()
    main()