import torch, numpy, nncore.ops, json, os, re, tqdm
from transformers import Qwen2VLProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import torch.distributed as dist

# CKPT = '/home/fangbenhao/L/models/Qwen2.5-VL-7B-Instruct'
# CKPT = '/home/fangbenhao/L/ETBench/cusmodel/Qwen2.5-VL-7B-Instruct-ETBENCH-custom_A/checkpoint-1250'
# CKPT = '/home/fangbenhao/L/ETBench/cusmodel/Qwen2.5-VL-7B-Instruct-ETBENCH-custom_B/checkpoint-1294'

CKPT = '/home/fangbenhao/L/models/Qwen2.5-VL-3B-Instruct'


DATA_PATH = '/home/fangbenhao/L/ETBench/etchat/eval/test_2.json'
SUBSETS = ['qvhighlights', 'thumos14', 'thumos15']
VIDEO_PATH = '/home/fangbenhao/L/DatasetS/ET_Data'
SUFFIX = '-et-bench-train-test2-'

# DATA_PATH = '/nas-wulanchabu/fuwen.luo/Open-R1-Video/data/et_bench/test_in_domain.json'
# VIDEO_PATH = '/nas-wulanchabu/fuwen.luo/datasets/et-bench/videos'
# SUFFIX = '-et-bench-in-domain'

OUTPUT = f'{CKPT}{SUFFIX}.json'

# dist.destroy_process_group()

FRAME_NUM = 16
# 401408 200704 None
MAX_PIXELS = 200704
TEMPLATE = [
    (
        """\
        You are a Vision Language Model specialized in interpreting visual data from chart images.
        Your task is to analyze the provided chart image and respond to queries with concise answers, usually a single word, number, or short phrase.
        The charts include a variety of types (e.g., line charts, bar charts) and contain colors, labels, and text.
        Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary.
        Reply with numbers pair only. Reply with numbers pair only.
        """
    ),
][0]
SAMPLING = False
THRS = [0.1, 0.3, 0.5, 0.7]

BATCH_NUM = 8

def main():
    try:
        with open(OUTPUT, 'r') as fp:
            data = json.load(fp)
        if 'video' not in data[-1].keys():
            data = data[:-1]
    except:
        with open(DATA_PATH, 'r') as fp:
            data = json.load(fp)

    model = LLM(
        model=CKPT,
        max_model_len=4096,
        max_num_seqs=BATCH_NUM,
        mm_processor_kwargs=({'max_pixels': MAX_PIXELS} if MAX_PIXELS is not None else {})

    )
    if MAX_PIXELS is None:
        processor = Qwen2VLProcessor.from_pretrained(CKPT)
    else:
        processor = Qwen2VLProcessor.from_pretrained(CKPT, max_pixels=MAX_PIXELS)

    batched_data = list()
    for entry in data:
        if 'result' not in entry.keys():
            if len(batched_data) == 0 or len(batched_data[-1]) >= BATCH_NUM:
                batched_data.append(list())
            batched_data[-1].append(entry)

    entry_ptr = 0
    for batch in tqdm.tqdm(batched_data):
        conversations, frames_selected = list(), list()
        for entry in batch:
            video_path = os.path.join(VIDEO_PATH, entry['video'])
            # make input, give the question
            frames_selected.append(numpy.linspace(0, entry['duration'], FRAME_NUM))
            frame_files = process_vision_info([{'content': [{'type': 'video', 'video': video_path, 'nframes': FRAME_NUM}]}])[1][0]
            # print(entry)
            entry['question'] = entry['conversations'][0]['value']
            conversations.append([{
                'role': 'user',
                'content': [
                    { 'type': 'video', 'video': frame_files },
                    { 'type': 'text', 'text': TEMPLATE.format(entry['question']) },
                ],
            }])

        # inference
        with torch.no_grad():
            text = [processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True) for conversation in conversations]
            inp = [{'prompt': item, 'multi_modal_data': {'video': conversation[0]['content'][0]['video']}} for item, conversation in zip(text, conversations)]
            if SAMPLING == True:
                generation_config = SamplingParams(top_p=0.5, top_k=250, temperature=1.0, max_tokens=512, stop_token_ids=None)
            else:
                generation_config = SamplingParams(top_p=1.0, top_k=1, temperature=0, max_tokens=512, stop_token_ids=None)
            out = model.generate(inp, sampling_params=generation_config)
            outputs = [o.outputs[0].text for o in out]




if __name__ == '__main__':
    main()

    if dist.is_initialized():
        dist.destroy_process_group()
