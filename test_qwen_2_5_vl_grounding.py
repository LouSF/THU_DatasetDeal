import torch, numpy, nncore.ops, json, os, re, tqdm
from transformers import Qwen2VLProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

CKPT = '/nas-wulanchabu/fuwen.luo/Open-R1-Video/Qwen-2.5-VL-7B-Video-GRPO-0312-2/checkpoint-1667'

DATA_PATH = '/nas-wulanchabu/fuwen.luo/Open-R1-Video/data/et_bench/test_train.json'
SUBSETS = ['qvhighlights', 'thumos14', 'thumos15']
VIDEO_PATH = '/nas-wulanchabu/fuwen.luo/datasets/et-bench/videos'
SUFFIX = '-et-bench-train'

# DATA_PATH = '/nas-wulanchabu/fuwen.luo/Open-R1-Video/data/et_bench/test_in_domain.json'
# VIDEO_PATH = '/nas-wulanchabu/fuwen.luo/datasets/et-bench/videos'
# SUFFIX = '-et-bench-in-domain'

OUTPUT = f'{CKPT}{SUFFIX}.json'

FRAME_NUM = 16
# 401408 200704 None
MAX_PIXELS = 200704
TEMPLATE = [
    (
        '{} Please select segments from the video which contain answer to the question. '
        'First, output reasoning process in <think> </think> tags. The reasoning process must REFER TO SPECIFIC TIMESTAMPS TO TELL WHERE YOU GET THE INFORMATION FROM THE VIDEO. '
        'Then summarize your reasoning process above and output selected segments like \"<segment>X.XX-X.XX</segment>\", where \"X\" denotes arabic numbers. If there are multiple segments, separate them with spaces like \"<segment>X.XX-X.XX X.XX-X.XX</segment>\". '
        'Your output format should be like \"<think>...</think><segment>...</segment>\".'
    ),
][0]
SAMPLING = False
THRS = [0.1, 0.3, 0.5, 0.7]

BATCH_NUM = 8


def eval(data):
    res_iou, m_iou, res_f1 = [0] * len(THRS), 0, [0] * len(THRS)
    for (segs, segs_sel) in data:
        # iou
        time_stamps_all = sorted([item_ for item in segs for item_ in item] + [item_ for item in segs_sel for item_ in item])
        union, intersection = 0, 0
        for i in range(len(time_stamps_all) - 1):
            overlap_counter = 0
            for seg in segs:
                if time_stamps_all[i] >= seg[0] and time_stamps_all[i + 1] <= seg[1]:
                    overlap_counter += 1
                    break
            for seg_sel in segs_sel:
                if time_stamps_all[i] >= seg_sel[0] and time_stamps_all[i + 1] <= seg_sel[1]:
                    overlap_counter += 1
                    break
            if overlap_counter >= 1:
                union += time_stamps_all[i + 1] - time_stamps_all[i]
            if overlap_counter >= 2:
                intersection += time_stamps_all[i + 1] - time_stamps_all[i]
        for i, thr in enumerate(THRS):
            if intersection / union >= thr:
                res_iou[i] += 1
        m_iou += intersection / union
        # f1
        if len(segs_sel) != 0:
            iou = nncore.ops.temporal_iou(torch.Tensor(segs), torch.Tensor(segs_sel))
            for i, thr in enumerate(THRS):
                if iou.max() < thr:
                    continue
                else:
                    rec = (iou.amax(dim=1) >= thr).float().mean().item()
                    prc = (iou.amax(dim=0) >= thr).float().mean().item()
                    res_f1[i] += 2 * prc * rec / (prc + rec)

    stat = { 'tot': len(data) }
    stat['m_iou'] = m_iou / len(data)
    for i, thr in enumerate(THRS):
        stat[f'iou_{thr}'] = res_iou[i] / len(data)
    stat['mean_iou'] = sum(res_iou) / len(data) / len(res_iou)
    for i, thr in enumerate(THRS):
        stat[f'f1_{thr}'] = res_f1[i] / len(data)
    stat['mean_f1'] = sum(res_f1) / len(data) / len(res_f1)
    return stat
def get_stat(data):
    stat = dict()
    for subset in SUBSETS:
        stat[subset] = eval([[entry[1], entry[2]] for entry in data if (subset in entry[0])])
    stat['all'] = eval([[entry[1], entry[2]] for entry in data])
    return stat


def main():
    try:
        with open(OUTPUT, 'r') as fp:
            data = json.load(fp)
        if 'video' not in data[-1].keys():
            data = data[:-1]
    except:
        with open(DATA_PATH, 'r') as fp:
            data = json.load(fp)

    model = LLM(model=CKPT, max_model_len=4096, max_num_seqs=BATCH_NUM, mm_processor_kwargs=({'max_pixels': MAX_PIXELS} if MAX_PIXELS is not None else {}))
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

        # get answer
        for conversation, output, frames_selected_ in zip(conversations, outputs, frames_selected):
            # conversation
            conversation.append({'role': 'assistant', 'content': [{'type': 'text', 'text': output}]})
            for msg in conversation:
                msg['content'] = ''.join([item['text'] for item in msg['content'] if item['type'] == 'text'])
            # selected segments
            try:
                segments_selection = list()
                assert len(re.findall(r'<segment>', output, re.DOTALL)) == 1
                assert len(re.findall(r'</segment>', output, re.DOTALL)) == 1
                for item in re.search(r'<segment>(.*)</segment>', output, re.DOTALL).group(1).strip().split():
                    t = item.split('-')
                    st, ed = float(t[0]), float(t[1])
                    assert st >= 0 and ed >= 0 and st < ed
                    segments_selection.append([st, ed])
            except:
                pass
            # skip processed entries
            while 'result' in data[entry_ptr].keys():
                entry_ptr += 1
            # save answer
            data[entry_ptr]['result'] = {
                'segments_selection': segments_selection,
                'frames': json.dumps(frames_selected_.tolist()),
                'conversations': [(msg['role'] + ': ' + msg['content']) for msg in conversation]
            }

        with open(OUTPUT, 'w') as fp:
            json.dump(data, fp, indent=4)

    data.append(get_stat([[entry['video'], entry['segments'], entry['result']['segments_selection']] for entry in data]))
    with open(OUTPUT, 'w') as fp:
        json.dump(data, fp, indent=4)


if __name__ == '__main__':
    main()
