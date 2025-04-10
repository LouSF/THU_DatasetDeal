import os
import json
import random

with open("et_instruct_164k_txt.json", "r") as f:
    all_data = json.load(f)

[random.shuffle(all_data) for _ in range(10)]

names = [10, 9, 8, 7, 6, 5, 4, 3]

sumC = {
    10: [200, 30, 70, 50],
    9: [300, 40, 60, 100],
    8: [450, 50, 50, 150],
    7: [700, 50, 50, 200],
    6: [1000, 100, 100, 300],
    5: [1100, 50, 100, 350],
    4: [2300, 100, 100, 500],
    3: [800, 100, 100, 500],
}

for name in names:
    l_10 = [i for i in all_data if i['task'] == 'tal' and 'tgt' in i and len(i['tgt']) == name * 2]

    [random.shuffle(l_10) for _ in range(10)]

    l_deal = []
    startC = 0
    for C in sumC[name]:
        endC = startC + C
        l_deal.append(l_10[startC:endC])
        startC = endC


    def change_A(conv, tgt):
        dict_words = ['zero', 'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth',
                      'tenth']
        # 确保随机范围有效
        max_idx = len(tgt) // 2 - 2
        if max_idx < 2:  # 如果范围太小，直接取中间值
            sum_idx = len(tgt) // 4  # 取近似中间位置
        else:
            sum_idx = random.randint(2, max_idx)  # 否则随机选择

        human_value = f"Please provide only the {dict_words[sum_idx]} specified time periods, along with the time period immediately before it (the {dict_words[sum_idx - 1]} one) and the time period immediately after it (the {dict_words[sum_idx + 1]} one), listed strictly in chronological order according to their start times; exclude all other time periods and any additional information."

        ai_value = f"The event happens in {tgt[sum_idx * 2]} - {tgt[sum_idx * 2 + 1]} seconds, {tgt[(sum_idx - 1) * 2]} - {tgt[(sum_idx - 1) * 2 + 1]} seconds, and {tgt[(sum_idx + 1) * 2]} - {tgt[(sum_idx + 1) * 2 + 1]} seconds."

        truth_conversations = [
            {
                "from": "human",
                "value": conv[0]['value'] + ' ' + human_value
            },
            {
                "from": "gpt",
                "value": ai_value
            }
        ]

        truth_tgt = [
            tgt[sum_idx * 2], tgt[sum_idx * 2 + 1],
            tgt[(sum_idx - 1) * 2], tgt[(sum_idx - 1) * 2 + 1],
            tgt[(sum_idx + 1) * 2], tgt[(sum_idx + 1) * 2 + 1],
        ]

        return truth_conversations, truth_tgt


    def change_B(conv, tgt):
        dict_words = ['zero', 'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth',
                      'tenth']
        max_idx = len(tgt) // 2 - 2
        if max_idx < 2:
            sum_idx_s = len(tgt) // 4
            sum_idx_e = len(tgt) // 4
        else:
            sum_idx_s = random.randint(2, max_idx)
            sum_idx_e = random.randint(2, max_idx)

        human_value = f"Please provide only the time periods corresponding to the {dict_words[sum_idx_s]} last segment and the {dict_words[sum_idx_e]} first segment, listed strictly in chronological order based on their start times, and exclude all other time periods or additional information."

        ai_value = f"The event happens in {tgt[sum_idx_s * 2]} - {tgt[sum_idx_s * 2 + 1]} seconds, and {tgt[sum_idx_e * 2]} - {tgt[sum_idx_e * 2 + 1]} seconds."

        truth_conversations = [
            {
                "from": "human",
                "value": conv[0]['value'] + human_value
            },
            {
                "from": "gpt",
                "value": ai_value,
            }
        ]

        truth_tgt = [
            tgt[sum_idx_s * 2], tgt[sum_idx_s * 2 + 1],
            tgt[sum_idx_e * 2], tgt[sum_idx_e * 2 + 1],
        ]

        return truth_conversations, truth_tgt


    def change_C(conv, tgt):
        human_value = "Given a list of video segment timestamps, return ONLY the timestamps of every even-numbered segment in order (i.e., the 2nd, 4th, 6th... entries in the original list). Do not modify the original timestamps or add any explanations."

        truth_conversations = [
            {
                "from": "human",
                "value": conv[0]['value'] + human_value
            },
            {
                "from": "gpt",
                "value": conv[1]['value']
            }
        ]

        truth_tgt = []
        for i in range(1, len(tgt) // 2, 2):
            segment_index = i * 2
            truth_tgt.append(tgt[segment_index])
            truth_tgt.append(tgt[segment_index + 1])

        return truth_conversations, truth_tgt


    for i in l_deal[0]:
        truth_conversations, truth_tgt = change_A(i['conversations'], i['tgt'])
        i['truth_conversations'] = truth_conversations
        i['truth_tgt'] = truth_tgt

    for i in l_deal[1]:
        truth_conversations, truth_tgt = change_B(i['conversations'], i['tgt'])
        i['truth_conversations'] = truth_conversations
        i['truth_tgt'] = truth_tgt

    for i in l_deal[2]:
        truth_conversations, truth_tgt = change_C(i['conversations'], i['tgt'])
        i['truth_conversations'] = truth_conversations
        i['truth_tgt'] = truth_tgt

    for i in l_deal[3]:
        i['truth_conversations'] = i['conversations']
        i['truth_tgt'] = i['tgt']

    # 修正这里：使用列表拼接而不是extend
    l_10_all = []
    for sublist in l_deal:
        l_10_all.extend(sublist)

    # 确保输出目录存在
    os.makedirs("new_comb", exist_ok=True)

    with open(f"new_comb/et_instruct_164k_txt_{name}.json", "w") as f:
        json.dump(l_10_all, f, indent=4)