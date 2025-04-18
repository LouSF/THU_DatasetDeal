import os
import re
import sys
import random
import json
import multiprocessing
import tqdm

import spacy
from lemminflect import getLemma, getInflection

json_path = "/Users/lsf/PycharmProjects/DatasetJson/NSjsondeal/Complise/star"
save_path = "/Users/lsf/PycharmProjects/DatasetJson/NSjsondeal/Complise/star/save"
json_file_list = os.listdir(json_path)
json_file_list = [rec for rec in json_file_list if rec.endswith(".json") and not rec.endswith("fix.json")]
json_label = json_path.split('/')[-1]

select_type = ["Sequence", "Prediction", "Interaction",]

select_options = {
    'v': [],
    'n': [],
}

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
    # s time start
    # e time end
    # b after or before
    # o ving+obj
    # j ved+obj
    # v base v
    # d past v
    # n n
    #k keep
    "Sequence": [
        [
            ("During {} to {}, the person did two things. Please list them sequentially.", 'se', 'S00'),
            ("What did the person do from {} to {}? List the things they do sequentially.", 'se', 'S01'),
            ("The person did A {} B between {} and {}. What are A and B?", 'bse', 'S02'),
            ("Focus on the segment {} - {}. What did the person do {} they {}?", 'sebj', 'S03'),
            ("Answer the above question according to the video. Only use words from the following words to organize your answer.", 'S04',),
        ],
        [
            ("Which object did the person {} {} they {} the ___ ? The ___.", 'vbd', 'S10'),
            ("The person ___ the ___ {} they ___ the ___.", 'b', 'S11'),
            ("Choose words from the following words to fill in the blanks according to segment {} - {} of the video.", 'se', 'S12'),
        ]
    ],
    "Prediction":[
        ("What will the person do after {} - {}?", "se", 'P0'),
        ("What will the person do next with the {} after {} - {}?", "nse", 'P1'),
        ("Which object would the person {} after {} - {}?", "vse", 'P2'),
        ("According to {} - {}, which object would the person {} next after they {} the {}?", "sekkk", 'P3'),
        ("Choose answer from the following options.", "", 'P4'),
    ]

}


def get_verb_forms(verb_phrase):
    words = verb_phrase.split()

    if not words:
        return {}

    # 只对第一个词进行变位
    verb = words[0]
    other_words = words[1:] if len(words) > 1 else []

    try:
        base = getLemma(verb, upos='VERB')[0]
        past = getInflection(base, tag='VBD')[0]
        past_part = getInflection(base, tag='VBN')[0]
        pres_part = getInflection(base, tag='VBG')[0]
    except (IndexError, TypeError):
        # 如果变位失败，使用原形
        base = verb
        past = verb
        past_part = verb
        pres_part = verb

    # 组合非动词部分
    other_part = ' '.join(other_words) if other_words else ''

    return {
        'base': f"{base} {other_part}".strip(),
        'past': f"{past} {other_part}".strip(),
        'past_participle': f"{past_part} {other_part}".strip(),
        'present_participle': f"{pres_part} {other_part}".strip()
    }

def generate_question_templates_Prediction(rec: dict):
    target_id = rec["question_id"].split('_')
    target_template = prompt_mop[target_id[0]][target_id[1]]

    time_index = 'before' if 'before' in rec["question"] else 'after'

    match = re.search(target_template, rec["question"])
    if not match:
        raise ValueError(
            f"Template '{target_template}' not found in question: '{rec["question"]}'"
        )
    matched_groups = match.groups() if match.lastindex else ()

    if len(matched_groups) == 0:
        create_option = "N"
    elif len(matched_groups) == 1:
        if "Which object would the person" in rec["question"]:
            create_option = "QV"
        else:
            create_option = "QO"
    else: # 3
        create_option = "Q+A"

    answers_list = []

    if create_option == "N":
        choice_group = prompt_target[target_id[0]][0]
        answers_list.append(rec["answer"].lower())
    elif create_option == "QV":
        choice_group = prompt_target[target_id[0]][2]
        answers_list.append(rec["answer"].lower())
    elif create_option == "QO":
        choice_group = prompt_target[target_id[0]][1]
        answers_list.append(rec["answer"].lower())
    else:
        choice_group = prompt_target[target_id[0]][3]
        answers_list.append(rec["answer"].lower())


    fixed_question = ""

    answers_list = [_.capitalize() for _ in answers_list]

    choice_prompt = choice_group[0]

    func_list = []
    ind = 0
    for func in choice_group[1]:
        if func == 's':
            func_list.append(rec["start"])
        elif func == 'e':
            func_list.append(rec["end"])
        elif func == 'v':
            func_list.append(matched_groups[0])
        elif func == 'n':
            func_list.append(matched_groups[0])
        elif func == 'b':
            func_list.append(time_index)
        elif func == 'o':
            func_list.append(answers_list[0])
            answers_list = [answers_list[1]]
        elif func == 'k':
            func_list.append(get_verb_forms(matched_groups[ind])['base'])
            ind += 1
    if len(func_list) != 0:
        fixed_question = choice_prompt.format(*func_list)

    choice_group = prompt_target[target_id[0]][-1]
    choice_prompt = choice_group[0]

    func_list = []
    for func in choice_group[1]:
        if func == 's':
            func_list.append(rec["start"])
        elif func == 'e':
            func_list.append(rec["end"])
    if len(func_list) != 0:
        choice_prompt = choice_prompt.format(*func_list)

    fixed_question = "".join([fixed_question, choice_prompt])

    fixed_options = [item["choice"] for item in rec["choices"]]

    for index, item in enumerate(answers_list):
        if item[-1] != '.':
            answers_list[index] = item + '.'

    fixed_options.extend(answers_list)
    fixed_options = list(set(fixed_options))
    random.shuffle(fixed_options)

    return fixed_question, fixed_options, answers_list, choice_group[-1]

def generate_question_templates_Sequence(rec: dict):
    target_id = rec["question_id"].split('_')
    target_template = prompt_mop[target_id[0]][target_id[1]]

    time_index = 'before' if 'before' in rec["question"] else 'after'

    match = re.search(target_template, rec["question"])
    if not match:
        raise ValueError(
            f"Template '{target_template}' not found in question: '{rec["question"]}'"
        )
    matched_groups = match.groups() if match.lastindex else ()

    # random_question_id = random.randint(0, prompt_target[target_id[0]] - 1)
    # prompt_target_choice = prompt_target[target_id[0]][random_question_id]
    prompt_target_choice = random.choice(prompt_target[target_id[0]])

    create_option = "Q+A" if len(matched_groups) > 2 else ""
    if create_option == "Q+A" and len(prompt_target_choice) == 3:
        create_option = "Q+A+V"

    answers_list = []

    # if create_option == "Q+A":
    #     answers_list.append((get_verb_forms(matched_groups[0])['ing'] + " " + rec["answer"]).lower())
    #     answers_list.append((get_verb_forms(matched_groups[1])['ing'] + " " + matched_groups[2]).lower())
    # else:
    #     answers_list.append((get_verb_forms(matched_groups[0])['ing'] + " " + matched_groups[1]).lower())
    #     answers_list.append((get_verb_forms(rec["answer"][:-1])['ing'] + " " + matched_groups[2]).lower())

    if create_option == "Q+A":
        answers_list.append((get_verb_forms(matched_groups[0])['present_participle'] + " " + rec["answer"]).lower())
        answers_list.append((get_verb_forms(matched_groups[1])['present_participle'] + " the " + matched_groups[2]).lower())

    elif create_option == "Q+A+V":
        answers_list.append((get_verb_forms(matched_groups[0])['past'] + " " + rec["answer"]).lower())
        answers_list.append((get_verb_forms(matched_groups[1])['past'] + " the " + matched_groups[2]).lower())

    else:
        answers_list.append((get_verb_forms(matched_groups[0])['present_participle'] + " the " + matched_groups[1]).lower())
        # temp_answer_tamplate = rec["answer"].split(" the ")
        # answers_list.append((temp_answer_tamplate[0] + " the " + temp_answer_tamplate[2]).lower())
        answers_list.append(rec["answer"].lower())



    fixed_question = ""

    answers_list = [_.capitalize() for _ in answers_list]

    choice_group = random.choice(prompt_target_choice[:-1])
    choice_prompt = choice_group[0]

    func_list = []
    for func in choice_group[1]:
        if func == 's':
            func_list.append(rec["start"])
        elif func == 'e':
            func_list.append(rec["end"])
        elif func == 'b':
            func_list.append(time_index)
        elif func == 'o':
            temp_ans = get_verb_forms(answers_list[0])['present_participle'].lower()
            func_list.append(temp_ans if temp_ans[-1] != '.' else temp_ans[:-1])
            answers_list = [get_verb_forms(answers_list[1])["present_participle"]]
        elif func == 'j':
            temp_ans = get_verb_forms(answers_list[0])['past'].lower()
            func_list.append(temp_ans if temp_ans[-1] != '.' else temp_ans[:-1])
            answers_list = [answers_list[1]]
        elif func == 'd':
            temp_ans = get_verb_forms(answers_list[0].split()[0])['past'].lower()
            func_list.append(temp_ans if temp_ans[-1] != '.' else temp_ans[:-1])
            answers_list = [answers_list[-1]]
        elif func == 'v':
            temp_ans = get_verb_forms(answers_list[0].split()[0])['base'].lower()
            func_list.append(temp_ans if temp_ans[-1] != '.' else temp_ans[:-1])


    if len(func_list) != 0:
        fixed_question = choice_prompt.format(*func_list)


    choice_group = prompt_target_choice[-1]
    choice_prompt = choice_group[0]

    func_list = []
    for func in choice_group[1]:
        if func == 's':
            func_list.append(rec["start"])
        elif func == 'e':
            func_list.append(rec["end"])
    if len(func_list) != 0:
        choice_prompt = choice_prompt.format(*func_list)

    fixed_question = "".join([fixed_question, ' ', choice_prompt])

    fixed_options = [item["choice"] for item in rec["choices"]]

    for index, item in enumerate(answers_list):
        if item[-1] != '.':
            answers_list[index] = item + '.'

    fixed_options.extend(answers_list)
    fixed_options = list(set(fixed_options))
    random.shuffle(fixed_options)

    return fixed_question, fixed_options, answers_list, choice_group[-1]



def main():

    # read files
    json_files_dict = {}
    for json_file_name in tqdm.tqdm(json_file_list, total=len(json_file_list), desc="Processing json files"):
        with open(os.path.join(json_path, json_file_name), 'r') as file:
            json_files_dict[json_file_name] = json.load(file)

    for json_file_name, json_data in tqdm.tqdm(json_files_dict.items(), total=len(json_files_dict), desc="Processing json files"):
        for rec in tqdm.tqdm(json_data, total=len(json_data), desc="Processing json files"):
            for _ in rec["choices"]:
                if _["choice"].startswith("The"):
                    select_options['n'].append(_["choice"])
                else:
                    select_options['v'].append(_["choice"])


    # split useful data
    for index, json_list in tqdm.tqdm(json_files_dict.items(), total=len(json_files_dict), desc="Processing json files"):
        json_files_dict[index] = [
            rec for rec in json_list
            if rec['question_id'].split('_')[0] in select_type
        ]


    # with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:

    #     pass

    # change dataset
    fixed_json_files_dict = {}
    for index, json_list in json_files_dict.items():
        fixed_json_list = []
        for rec in tqdm.tqdm(json_list, total=len(json_list), desc=f"Processing {index}"):
            # try:
                fixed_question, fixed_options, fixed_answers = [], [], []
                if rec['question_id'].split('_')[0] == "Sequence":
                    fixed_question, fixed_options, fixed_answers, types = generate_question_templates_Sequence(rec)
                elif rec['question_id'].split('_')[0] == "Prediction":
                    fixed_question, fixed_options, fixed_answers, types = generate_question_templates_Prediction(rec)
                elif rec['question_id'].split('_')[0] == "Interaction":
                    fixed_question, fixed_options, fixed_answers, types = (
                        "".join([rec["question"], " ",  f"Choose answer from the following options according to fragment {rec["start"]} - {rec["end"]} of the video."]),
                        [item["choice"] for item in rec["choices"]],
                        rec["answer"], 'I0',
                    )

                fixed_options += random.choices(select_options['n' if fixed_answers[0].startswith('The') else 'v'], k=random.randint(3, 5))
                fixed_options = list(set(fixed_options))

                fixed_json_list.append(
                    {
                        "dataset": json_label,
                        "task": rec['question_id'].split('_')[0],
                        "fixed_type": types,
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
                                "value": fixed_answers,
                            },
                            # {
                            #     "from": "gpt",
                            #     "type": "text",
                            #     "value": fixed_answers["respond"],
                            # }
                        ],
                        "options": fixed_options,
                    }
                )
            # except Exception as e:
            #     print(e)

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