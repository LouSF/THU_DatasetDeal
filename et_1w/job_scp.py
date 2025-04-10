import re
from typing import Optional
import random
from optparse import Option

dict_words = ['zero', 'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth',
                      'tenth']

pattern = re.compile(r"'([^']*)'")

def job_0(input_rec: dict) -> dict:
    '''
    给出所有段的时间（不变）
    '''

    truth_conversations = input_rec['conversations']
    truth_tgt = input_rec['tgt']
    input_rec.update(
        {
            'truth_conversations': truth_conversations,
            'truth_tgt': truth_tgt,
            'method': 'job_0'
         }
    )
    return input_rec


def job_1(input_rec: dict) -> dict:
    '''
    给出第X段的时间及其后一段时间（2段）
    '''

    tgt = input_rec['tgt']
    max_idx = len(tgt) // 2 - 2
    if max_idx < 2:
        sum_idx = len(tgt) // 4
    else:
        sum_idx = random.randint(2, max_idx)

    # human_value = f'Please provide only the {dict_words[sum_idx]} specified time periods, along with the time period immediately after it (the {dict_words[sum_idx + 1]} one).'

    pattern_value = pattern.search(input_rec['conversations'][0]['value']).group()
    human_value = [
        f"Identify the {dict_words[sum_idx + 1]} video segment aligned with query {pattern_value}, along with the segment immediately following it which is also aligned with the query.",
        f"Retrieve the {dict_words[sum_idx + 1]} occurrence of {pattern_value} along with the following segment that also fulfills the query.",
        f"List the {dict_words[sum_idx + 1]} segment matching the query {pattern_value} combined with its direct successor in timeline which also matches the query.",
    ]
    human_value = random.choice(human_value)

    print(human_value)

    ai_value = f'The event happens in {tgt[sum_idx * 2]} - {tgt[sum_idx * 2 + 1]} seconds, and {tgt[(sum_idx + 1) * 2]} - {tgt[(sum_idx + 1) * 2 + 1]} seconds.'
    truth_conversations = [
        {
            "from": "human",
            # "value": rec['conversations'][0]['value'] + human_value
            "value": human_value
        },
        {
            "from": "gpt",
            "value": ai_value
        }
    ]

    truth_tgt = [
        tgt[sum_idx * 2], tgt[sum_idx * 2 + 1],
        tgt[(sum_idx + 1) * 2], tgt[(sum_idx + 1) * 2 + 1],
    ]

    input_rec.update(
        {
            'truth_conversations': truth_conversations,
            'truth_tgt': truth_tgt,
            'method': 'job_1'
         }
    )

    return input_rec


def job_2(input_rec: dict) -> dict:
    '''
    给出第X段的时间及其前一段时间（2段）
    '''

    tgt = input_rec['tgt']
    max_idx = len(tgt) // 2 - 2
    if max_idx < 2:
        sum_idx = len(tgt) // 4
    else:
        sum_idx = random.randint(2, max_idx)

    # human_value = f'Please provide only the {dict_words[sum_idx]} specified time periods, along with the time period immediately before it (the {dict_words[sum_idx - 1]} one).'

    pattern_value = pattern.search(input_rec['conversations'][0]['value']).group()
    human_value = [
        f"Identify and list the {dict_words[sum_idx + 1]} segment of the video that aligns with the query {pattern_value}, along with the segment immediately preceding it which is also aligned with the query.",
        f"Retrieve the {dict_words[sum_idx + 1]} occurrence of {pattern_value} along with the preceding segment that also fulfills the query.",
        f"List the {dict_words[sum_idx + 1]} segment matching the query {pattern_value} combined with its direct predecessor in timeline which also matches the query.",
    ]
    human_value = random.choice(human_value)

    print(human_value)

    ai_value = f'The event happens in {tgt[sum_idx * 2]} - {tgt[sum_idx * 2 + 1]} seconds, and {tgt[(sum_idx - 1) * 2]} - {tgt[(sum_idx - 1) * 2 + 1]} seconds.'

    truth_conversations = [
        {
            "from": "human",
            # "value": rec['conversations'][0]['value'] + human_value
            "value": human_value
        },
        {
            "from": "gpt",
            "value": ai_value
        }
    ]

    truth_tgt = [
        tgt[sum_idx * 2], tgt[sum_idx * 2 + 1],
        tgt[(sum_idx - 1) * 2], tgt[(sum_idx - 1) * 2 + 1],
    ]

    input_rec.update(
        {
            'truth_conversations': truth_conversations,
            'truth_tgt': truth_tgt,
            'method': 'job_2'
         }
    )

    return input_rec


def job_3(input_rec: dict) -> dict:
    '''
    给出第X段的时间及其前后一段的时间（3段）
    '''

    tgt = input_rec['tgt']
    max_idx = len(tgt) // 2 - 2
    if max_idx < 2:
        sum_idx = len(tgt) // 4
    else:
        sum_idx = random.randint(2, max_idx)

    # human_value = f'Please provide only the {dict_words[sum_idx]} specified time periods, along with the time period immediately before it (the {dict_words[sum_idx - 1]} one), and the time period immediately after it (the {dict_words[sum_idx + 1]} one).'

    pattern_value = pattern.search(input_rec['conversations'][0]['value']).group()
    human_value = [
        f"Identify and list the {dict_words[sum_idx + 1]} segment matching the query {pattern_value}, along with the immediately preceding and following segments that also align with the query (in total 3 segments).",
        f"Find the {dict_words[sum_idx + 1]} video segment matching {pattern_value}, along with the preceding and following segments that also fit the query (in total 3 segments).",
        f"Locate the {dict_words[sum_idx + 1]} video segment related to {pattern_value}, then retrieve the segment directly before it and the one directly after it that also meet the query. List 3 segments in total.",
    ]
    human_value = random.choice(human_value)

    print(human_value)

    ai_value = f'The event happens in {tgt[sum_idx * 2]} - {tgt[sum_idx * 2 + 1]} seconds, and {tgt[(sum_idx - 1) * 2]} - {tgt[(sum_idx - 1) * 2 + 1]} seconds, and {tgt[(sum_idx + 1) * 2]} - {tgt[(sum_idx + 1) * 2 + 1]} seconds.'
    truth_conversations = [
        {
            "from": "human",
            # "value": rec['conversations'][0]['value'] + human_value
            "value": human_value
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

    input_rec.update(
        {
            'truth_conversations': truth_conversations,
            'truth_tgt': truth_tgt
         }
    )

    return input_rec


def job_4(input_rec: dict) -> dict:
    '''
    给出<time>-<time>秒数后一段时间（1段）
    '''

    tgt = input_rec['tgt']
    max_idx = len(tgt) // 2 - 2
    if max_idx < 2:
        sum_idx = len(tgt) // 4
    else:
        sum_idx = random.randint(2, max_idx)

    # human_value = f'Given a video with multiple segments, please provide ONLY the time period immediately after the specified range of {tgt[sum_idx * 2]} - {tgt[sum_idx * 2 + 1]} seconds.'

    pattern_value = pattern.search(input_rec['conversations'][0]['value']).group()
    human_value = [
        f"Identify and list the segment immediately after {tgt[sum_idx * 2]} - {tgt[sum_idx * 2 + 1]} seconds which aligns with query {pattern_value}.",
        f"Locate the video segment following {tgt[sum_idx * 2]} - {tgt[sum_idx * 2 + 1]} seconds that matches the query {pattern_value}.",
        f"Detect the video clip just after {tgt[sum_idx * 2]} - {tgt[sum_idx * 2 + 1]} seconds corresponding to query {pattern_value}.",
    ]
    human_value = random.choice(human_value)

    print(human_value)

    ai_value = f'The event happens in {tgt[(sum_idx + 1) * 2]} - {tgt[(sum_idx + 1) * 2 + 1]} seconds.'
    truth_conversations = [
        {
            "from": "human",
            # "value": rec['conversations'][0]['value'] + human_value
            "value": human_value
        },
        {
            "from": "gpt",
            "value": ai_value
        }
    ]

    truth_tgt = [
        tgt[(sum_idx + 1) * 2], tgt[(sum_idx + 1) * 2 + 1],
    ]

    input_rec.update(
        {
            'truth_conversations': truth_conversations,
            'truth_tgt': truth_tgt,
            'method': 'job_4'
        }
    )

    return input_rec


def job_5(input_rec: dict) -> dict:
    '''
    给出<time>-<time>秒数前一段时间（1段）
    '''

    tgt = input_rec['tgt']
    max_idx = len(tgt) // 2 - 2
    if max_idx < 2:
        sum_idx = len(tgt) // 4
    else:
        sum_idx = random.randint(2, max_idx)

    # human_value = f'Given a video with multiple segments, please provide ONLY the time period immediately before the specified range of {tgt[sum_idx * 2]} - {tgt[sum_idx * 2 + 1]} seconds.'

    pattern_value = pattern.search(input_rec['conversations'][0]['value']).group()
    human_value = [
        f"Identify and list the segment immediately before {tgt[sum_idx * 2]} - {tgt[sum_idx * 2 + 1]} seconds which aligns with query {pattern_value}.",
        f"Locate the video segment just before {tgt[sum_idx * 2]} - {tgt[sum_idx * 2 + 1]} seconds that matches the query {pattern_value}.",
        f"Detect the video clip just before {tgt[sum_idx * 2]} - {tgt[sum_idx * 2 + 1]} seconds corresponding to query {pattern_value}.",
    ]
    human_value = random.choice(human_value)

    print(human_value)

    ai_value = f'The event happens in {tgt[(sum_idx - 1) * 2]} - {tgt[(sum_idx - 1) * 2 + 1]} seconds.'
    truth_conversations = [
        {
            "from": "human",
            # "value": rec['conversations'][0]['value'] + human_value
            "value": human_value
        },
        {
            "from": "gpt",
            "value": ai_value
        }
    ]

    truth_tgt = [
        tgt[(sum_idx - 1) * 2], tgt[(sum_idx - 1) * 2 + 1],
    ]

    input_rec.update(
        {
            'truth_conversations': truth_conversations,
            'truth_tgt': truth_tgt,
            'method': 'job_5'
         }
    )

    return input_rec


def job_6(input_rec: dict) -> dict:
    '''
    给出<time>-<time>秒数前后一段时间（2段）
    '''

    tgt = input_rec['tgt']
    max_idx = len(tgt) // 2 - 2
    if max_idx < 2:
        sum_idx = len(tgt) // 4
    else:
        sum_idx = random.randint(2, max_idx)

    # human_value = f'Given a video with segments, provide ONLY:  The time period immediately BEFORE {tgt[sum_idx * 2]} - {tgt[sum_idx * 2 + 1]} seconds, and the time period immediately AFTER it.'

    pattern_value = pattern.search(input_rec['conversations'][0]['value']).group()
    human_value = [
        f"Given a query {pattern_value} and a fragment {tgt[sum_idx * 2]} - {tgt[sum_idx * 2 + 1]} seconds, identify the fragments immediately before and after it that also agree with the query.",
        f"Given query {pattern_value}, locate the last segment before {tgt[sum_idx * 2]} seconds and the first segment after {tgt[sum_idx * 2 + 1]} seconds that correspond to the query.",
        f"Given query {pattern_value} and segment {tgt[sum_idx * 2]} - {tgt[sum_idx * 2 + 1]} seconds, find its preceding segment and subsequent segment which also match the query.",
    ]
    human_value = random.choice(human_value)

    print(human_value)

    ai_value = f'The event happens in {tgt[(sum_idx - 1) * 2]} - {tgt[(sum_idx - 1) * 2 + 1]} seconds, and {tgt[(sum_idx + 1) * 2]} - {tgt[(sum_idx + 1) * 2 + 1]} seconds.'
    truth_conversations = [
        {
            "from": "human",
            # "value": rec['conversations'][0]['value'] + human_value
            "value": human_value
        },
        {
            "from": "gpt",
            "value": ai_value
        }
    ]

    truth_tgt = [
        tgt[(sum_idx - 1) * 2], tgt[(sum_idx - 1) * 2 + 1],
        tgt[(sum_idx + 1) * 2], tgt[(sum_idx + 1) * 2 + 1],
    ]

    input_rec.update(
        {
            'truth_conversations': truth_conversations,
            'truth_tgt': truth_tgt,
            'method': 'job_6'
         }
    )

    return input_rec

def job_7(input_rec: dict) -> dict:
    '''
    给出正数第X段时间（1段）
    '''

    tgt = input_rec['tgt']
    max_idx = len(tgt) // 2 - 2
    if max_idx < 2:
        sum_idx = len(tgt) // 4
    else:
        sum_idx = random.randint(2, max_idx)

    # human_value = f'Extract ONLY the time range of the {dict_words[sum_idx]} segment (in chronological order).'

    pattern_value = pattern.search(input_rec['conversations'][0]['value']).group()
    human_value = [
        f"Identify the {dict_words[sum_idx]} segment which aligns with the query {pattern_value}.",
        f"Locate the {dict_words[sum_idx]} segment corresponding to the query {pattern_value}.",
        f"Find the {dict_words[sum_idx]} segment that matches the query {pattern_value}.",
    ]
    human_value = random.choice(human_value)

    print(human_value)

    ai_value = f'The event happens in {tgt[(sum_idx - 1) * 2]} - {tgt[(sum_idx - 1) * 2 + 1]} seconds.'
    truth_conversations = [
        {
            "from": "human",
            # "value": rec['conversations'][0]['value'] + human_value
            "value": human_value
        },
        {
            "from": "gpt",
            "value": ai_value
        }
    ]

    truth_tgt = [
        tgt[(sum_idx - 1) * 2], tgt[(sum_idx - 1) * 2 + 1],
    ]

    input_rec.update(
        {
            'truth_conversations': truth_conversations,
            'truth_tgt': truth_tgt,
            'method': 'job_7'
         }
    )

    return input_rec


def job_8(input_rec: dict) -> dict:
    '''
    给出倒数后X段时间（1段）
    '''

    tgt = input_rec['tgt']
    max_idx = len(tgt) // 2 - 2
    if max_idx < 2:
        sum_idx = len(tgt) // 4
    else:
        sum_idx = random.randint(2, max_idx)

    # human_value = f'Extract ONLY the time range of the {dict_words[sum_idx]} segment COUNTING FROM THE END (1=last segment).'

    pattern_value = pattern.search(input_rec['conversations'][0]['value']).group()
    human_value = [
        f"Identify the {dict_words[sum_idx] if sum_idx != 1 else ''} last segment which aligns with the query {pattern_value}.",
        f"Locate the {dict_words[sum_idx] if sum_idx != 1 else ''} last segment corresponding to the query {pattern_value}.",
        f"Find the {dict_words[sum_idx] if sum_idx != 1 else ''} last segment that matches the query {pattern_value}.",
    ]
    human_value = random.choice(human_value)
    human_value = re.sub(r'\s+', ' ', human_value).strip()

    print(human_value)

    ai_value = f'The event happens in {tgt[-sum_idx * 2]} - {tgt[-sum_idx * 2 + 1]} seconds.'
    truth_conversations = [
        {
            "from": "human",
            # "value": rec['conversations'][0]['value'] + human_value
            "value": human_value
        },
        {
            "from": "gpt",
            "value": ai_value
        }
    ]

    truth_tgt = [
        tgt[-sum_idx * 2], tgt[-sum_idx * 2 + 1],
    ]

    input_rec.update(
        {
            'truth_conversations': truth_conversations,
            'truth_tgt': truth_tgt,
            'method': 'job_8'
         }
    )

    return input_rec


def job_9(input_rec: dict) -> Optional[dict]:
    '''
    给出正数前X段时间（多段）
    '''

    tgt = input_rec['tgt']
    max_idx = len(tgt) // 2 - 2
    if max_idx < 2 + 1:
        sum_idx = len(tgt) // 4
        return None
    else:
        sum_idx = random.randint(2, max_idx - 1)

    truth_tgt = []
    for i in range(0, sum_idx):
        truth_tgt.extend([tgt[i * 2], tgt[i * 2 + 1], ])

    # human_value = f'Please provide only the first {dict_words[sum_idx]} specified time periods.'

    pattern_value = pattern.search(input_rec['conversations'][0]['value']).group()
    human_value = [
        f"Identify the first {sum_idx} segments which align with the query {pattern_value}.",
        f"Locate the initial {sum_idx} segments that correspond to the query {pattern_value}.",
        f"Find the first {sum_idx} segments matching the query {pattern_value}.",
    ]
    human_value = random.choice(human_value)

    print(human_value)

    ai_value = "The event happens in" + "".join(
        f", {truth_tgt[i]} - {truth_tgt[i + 1]} seconds"
        for i in range(0, len(truth_tgt), 2)
    ) + "."

    truth_conversations = [
        {
            "from": "human",
            # "value": rec['conversations'][0]['value'] + human_value
            "value": human_value
        },
        {
            "from": "gpt",
            "value": ai_value
        }
    ]

    input_rec.update(
        {
            'truth_conversations': truth_conversations,
            'truth_tgt': truth_tgt,
            'method': 'job_9'
         }
    )

    return input_rec


def job_10(input_rec: dict) -> Optional[dict]:
    '''
    给出倒数后X段时间（多段）
    '''
    tgt = input_rec['tgt']
    max_idx = len(tgt) // 2 - 2
    if max_idx < 2 + 1:
        # sum_idx = len(tgt) // 4
        return None
    else:
        sum_idx = random.randint(2 + 1, max_idx)

    truth_tgt = []
    for i in range(sum_idx - 1, -1, -1):
        truth_tgt.extend([tgt[i * 2], tgt[i * 2 + 1], ])

    # human_value = f'Please provide only the last {dict_words[sum_idx]} specified time periods.'

    pattern_value = pattern.search(input_rec['conversations'][0]['value']).group()
    human_value = [
        f"Identify the last {sum_idx if sum_idx != 1 else ''} segments which align with the query {pattern_value}.",
        f"Locate the final {sum_idx if sum_idx != 1 else ''} segments that correspond to the query {pattern_value}.",
        f"Determine the last {sum_idx if sum_idx != 1 else ''} segments matching the query {pattern_value}.",
    ]
    human_value = random.choice(human_value)
    human_value = re.sub(r'\s+', ' ', human_value).strip()

    print(human_value)

    ai_value = "The event happens in" + "".join(
        f", {truth_tgt[i]} - {truth_tgt[i + 1]} seconds"
        for i in range(0, len(truth_tgt), 2)
    ) + "."

    truth_conversations = [
        {
            "from": "human",
            # "value": rec['conversations'][0]['value'] + human_value
            "value": human_value
        },
        {
            "from": "gpt",
            "value": ai_value
        }
    ]

    input_rec.update(
        {
            'truth_conversations': truth_conversations,
            'truth_tgt': truth_tgt,
            'method': 'job_10'
         }
    )

    return input_rec


def job_11(input_rec: dict) -> dict:
    '''
    给出所有奇数/偶数段时间（多段）
    '''

    tgt = input_rec['tgt']
    max_idx = len(tgt) // 2 - 2
    if max_idx < 2:
        sum_idx = len(tgt) // 4
    else:
        sum_idx = random.randint(2, max_idx)

    truth_tgt = []
    for i in range(1, len(tgt) // 2, 2):
        segment_index = i * 2
        truth_tgt.append(tgt[segment_index])
        truth_tgt.append(tgt[segment_index + 1])

    human_value = f'Please provide only the {dict_words[sum_idx]} specified time periods.'
    ai_value = "The event happens in" + "".join(
        f", {truth_tgt[i]} - {truth_tgt[i + 1]} seconds"
        for i in range(0, len(truth_tgt), 2)
    ) + "."
    truth_conversations = [
        {
            "from": "human",
            # "value": rec['conversations'][0]['value'] + human_value
            "value": human_value
        },
        {
            "from": "gpt",
            "value": ai_value
        }
    ]

    input_rec.update(
        {
            'truth_conversations': truth_conversations,
            'truth_tgt': truth_tgt,
            'method': 'job_11'
         }
    )

    return input_rec



job_list = [job_0, job_1, job_2, job_3, job_4, job_5, job_6, job_7, job_8, job_9, job_10, job_11]

job_probabilities = [
    0.05,   # job_0
    0.25,   # job_1
    0.25,   # job_2
    0.05,   # job_3
    0.1,    # job_4
    0.1,    # job_5
    0.05,   # job_6
    0.05,   # job_7
    0.05,   # job_8
    0.025,  # job_9
    0.025,  # job_10
    0.00,   # job_11
]

def calculate_time_segments_ratio(duration, tgt):
    if len(tgt) % 2 != 0:
        raise ValueError("tgt list")

    total_segments_time = 0.0
    for i in range(0, len(tgt), 2):
        start = tgt[i]
        end = tgt[i + 1]
        segment_duration = end - start
        total_segments_time += segment_duration


    try:
        ratio = int(duration / total_segments_time)
    except:
        ratio = 999
    return ratio

# def calculate_time_segments_ratio(duration, tgt, method = 'max'):
#     if len(tgt) % 2 != 0:
#         raise ValueError("tgt list")
#
#     ratio_max = -1
#     ratio_min = 999
#     for i in range(0, len(tgt), 2):
#         start = tgt[i]
#         end = tgt[i + 1]
#         segment_duration = end - start
#         try:
#             ratio = int(duration / segment_duration)
#             ratio_max = max(ratio, ratio_max)
#             ratio_min = min(ratio, ratio_min)
#         except:
#             pass
#
#     return ratio_max if method == 'max' else ratio_min

def selcet_ratio(input_rec: list, method = 'raw') -> list:
    result = []
    for rec in input_rec:
        ratio = calculate_time_segments_ratio(rec['duration'], rec['tgt'], 'min')
        if ratio <= 16:
            rec.update(
                {
                    'ratio': ratio
                }
            )
            result.append(rec)
    return result

def selcet_ratio_ed(input_rec: list) -> list:
    result = []
    for rec in input_rec:
        ratio = calculate_time_segments_ratio(rec['duration'], rec['truth_tgt'], 'max')
        if ratio <= 16:
            rec.update(
                {
                    'ratio': ratio
                }
            )
            result.append(rec)
    return result


def add_sum():
    total = 0
    for ele in range(0, len(job_probabilities)):
        total += job_probabilities[ele]

    print(total)

if __name__ == '__main__':
    add_sum()