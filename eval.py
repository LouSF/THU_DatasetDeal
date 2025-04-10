import os
import re
import json
import torch
import nncore.ops

source_path = 'data_json'
target_path = 'results_data_json'
# source_path = 'js_A'
# target_path = 'js_B'
eval_path = 'eval'
SUBSETS = ['qvhighlights', 'thumos14', 'thumos15', 'charades_sta',] # 'perception_test',
TASK = ['tem', 'tal',]
THRS = [0.1, 0.3, 0.5, 0.7]


if not os.path.exists(target_path):
    os.makedirs(target_path)

if not os.path.exists(eval_path):
    os.makedirs(eval_path)

json_file_list = {path.split('.')[0]: os.path.join(source_path, path) for path in os.listdir(source_path)}

def eval(data):
    res_iou, m_iou, res_f1 = [0] * len(THRS), 0, [0] * len(THRS)
    for (segs, segs_sel) in data:
        # iou
        time_stamps_all = sorted([item_ for item in segs for item_ in item] + [item_ for item in segs_sel for item_ in item])
        # time_stamps_all = sorted([item_ for item in segs for item_ in item] + [item_ for item in segs_sel for item_ in item])

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
    try:
        stat['m_iou'] = m_iou / len(data)
    except ZeroDivisionError:
        stat['m_iou'] = 0

    for i, thr in enumerate(THRS):
        try:
            stat[f'iou_{thr}'] = res_iou[i] / len(data)
        except ZeroDivisionError:
            stat[f'iou_{thr}'] = 0

    try:
        stat['mean_iou'] = sum(res_iou) / len(data) / len(res_iou)
    except ZeroDivisionError:
        stat[f'mean_iou'] = 0

    for i, thr in enumerate(THRS):
        try:
            stat[f'f1_{thr}'] = res_f1[i] / len(data)
        except ZeroDivisionError:
            stat[f'f1_{thr}'] = 0

    try:
        stat['mean_f1'] = sum(res_f1) / len(data) / len(res_f1)
    except ZeroDivisionError:
        stat[f'mean_f1'] = 0

    return stat

def get_stat(data):
    stat = dict()

    for subset in SUBSETS:
        stat[subset] = eval([[entry[1], entry[2]] for entry in data if (subset in entry[0])])

    for task in TASK:
        stat["perception_test" +"_"+ task] = eval([[entry[1], entry[2]] for entry in data if (task in entry[-1] and "perception_test" in entry[0])])

    stat['all'] = eval([[entry[1], entry[2]] for entry in data])

    return stat

for name, json_file in json_file_list.items():
    with open(json_file, 'r') as fp:
        datas = json.load(fp)
        for data in datas:
            text = data['result']['conversations'][1]
            pattern = r'(\d+\.?\d*)\s*([-,~ ])\s*(\d+\.?\d*)'
            matches = re.findall(pattern, text)
            segments_selection = []
            for match in matches:
                start, _, end = match
                start, end = float(start), float(end)
                if end < start:
                    segments_selection.extend([end, start])
                else:
                    segments_selection.extend([start, end])

            tgt = data['tgt']

            interval_pairs = [tgt[i:i + 2] for i in range(0, len(tgt), 2)]
            segments_selections = [segments_selection[i:i+2] for i in range(0, len(segments_selection), 2)]
            data['result']['segments_selection'] = segments_selections
            data['segments'] = interval_pairs


        with (open(os.path.join(target_path, json_file.split('/')[1]), 'w')) as fp:
            json.dump(datas, fp, indent=4)


for name, json_file in json_file_list.items():
    with (open(os.path.join(target_path, json_file.split('/')[1]), 'r')) as fp:
        datas = json.load(fp)
        datas.append(
            get_stat([[entry['video'], entry['segments'], entry['result']['segments_selection'], entry['task']] for entry in datas])
        )

    with (open(os.path.join(eval_path, json_file.split('/')[1]), 'w')) as fp:
        json.dump(datas, fp, indent=4)