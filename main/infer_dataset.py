import os
import re
import tqdm
import json
import argparse
from datetime import date
from pathlib import Path

import numpy
import torch
from transformers import Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info

from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset
from vllm.utils import FlexibleArgumentParser

from openai import OpenAI

pixels_kwargs = 200704
frame_num = 16
batch_num = 8
gpu_num = 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_path')
    parser.add_argument('--data_path')
    parser.add_argument('--pred_path')
    parser.add_argument('--model_path')
    parser.add_argument('--chunk', type=int, default=1)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()

system_message = [
    "You are a Vision Language Model specialized in interpreting visual data from chart images. \
    Your task is to analyze the provided chart image and respond to queries with concise answers, usually a single word, number, or short phrase. \
    The charts include a variety of types (e.g., line charts, bar charts) and contain colors, labels, and text. \
    Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary.",

    "You are a helpful assistant.\
    The format of your response should be: 'The action happens in <start time> - <end time>, <start time> - <end time>, and <start time> - <end time>'.<|im_end|>"

]

def run_qwen2_5_vl(question: str):
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.The format of your response should be: 'The action happens in <start time> - <end time>, <start time> - <end time>, and <start time> - <end time>'.<|im_end|>\n"
        f"<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n")
    stop_token_ids = None
    return prompt, stop_token_ids

def build_messages(video_path, query):
    return [{
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": f"{video_path}",
                "max_pixels": pixels_kwargs,
                "fps": frame_num,
            },
            {"type": "text", "text": query}
        ]
    }]


if __name__ == '__main__':

    print("Predeal......")
    args = parse_args()
    if args.chunk > 1:
        pred_path = Path(args.pred_path) / f'etbench_{args.index}.json'
    else:
        pred_path = Path(args.pred_path) / f'etbench_Qwen2_5_VL_{str(date.today())}_all.json'
    print(f'Chunk: {args.chunk} Index: {args.index} Output Path: {pred_path}')

    data_path = args.data_path
    anno_path = args.anno_path

    print("Predeal succeed!")

    print("Loading model......")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=4,
        trust_remote_code=True,
        max_model_len=4096,
        max_num_seqs=gpu_num,
        mm_processor_kwargs={
            "use_fast": True,
            "max_pixels": pixels_kwargs,
            "fps": frame_num,
        },
        disable_mm_preprocessor_cache=False,
    )
    processor = Qwen2_5_VLProcessor.from_pretrained(
        args.model_path,
        max_pixels=pixels_kwargs,
        use_fast=True,
    )
    sampling_params = SamplingParams(
        top_p=1.0,
        top_k=1,
        temperature=0,
        max_tokens=512,
        stop_token_ids=None,
    )
    print("Loading model succeed!")

    print("Bactching dataset......")
    try:
        with open(pred_path, 'r') as fp:
            datas = json.load(fp)
        if 'video' not in datas[-1].keys():
            datas = datas[:-1]
    except:
        with open(anno_path, 'r') as fp:
            datas = json.load(fp)

    batched_data = []
    for data in datas:
        if 'result' not in data.keys():
            if len(batched_data) == 0 or len(batched_data[-1]) >= batch_num:
                batched_data.append([])
            batched_data[-1].append(data)
    print("Bactching dataset finished!")

    for batch in tqdm.tqdm(batched_data):
        conversations, frames_selected = [], []
        for data in batch:
            video_path = os.path.join(data_path, data['video'])
            # make input, give the question
            frames_selected.append(numpy.linspace(0, data['duration'], frame_num))
            frame_files = process_vision_info(
                [
                    {
                        'content': [
                            {
                                'type': 'video',
                                'video': video_path,
                                'nframes': frame_num,
                            }
                        ]
                    }
                ]
            )[1][0]
            data['question'] = data['conversations'][0]['value']
            conversations.append([
                {
                    'role': 'system',
                    'content': system_message,
                },
                {
                    'role': 'user',
                    'content': [
                        {'type': 'video', 'video': frame_files},
                        {'type': 'text', 'text': run_qwen2_5_vl(data['question'])[1]},
                    ],
                }
            ])

        with torch.no_grad():
            text = [
                processor.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=True
                ) for conversation in conversations
            ]
            inp = [
                {
                    'prompt': item,
                    'multi_modal_data': {'video': conversation[-1]['content'][0]['video']}
                }
                for item, conversation in zip(text, conversations)
            ]
            out = llm.generate(
                inp,
                sampling_params=sampling_params
            )
            outputs = [o.outputs[0].text for o in out]

        for data, output in zip(batch, outputs):
            data['a'] = output

            # if args.verbose:
            #     print()
            #     print(f"\nPrompt: {data['conversations'][0]['value']}")
            #     print(f"response: {output}")

        with open(pred_path, 'w') as fp:
            json.dump(batch, fp, indent=4)


    if dist.is_initialized():
        dist.destroy_process_group()