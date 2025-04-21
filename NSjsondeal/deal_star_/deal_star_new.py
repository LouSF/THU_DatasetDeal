import os
import re
import sys
import json
import tqdm
import random

import multiprocessing

import spacy
from lemminflect import getLemma, getInflection

from NSjsondeal import deal_star

json_path = "/Users/lsf/PycharmProjects/DatasetJson/NSjsondeal/Complise/star"
save_path = "/Users/lsf/PycharmProjects/DatasetJson/NSjsondeal/Complise/star/save"
json_file_list = os.listdir(json_path)
json_file_list = [rec for rec in json_file_list if rec.endswith(".json") and not rec.endswith("fix.json")]
json_label = json_path.split('/')[-1]

debug = False

select_options = {'verb': [], 'noun': [],}
json_files_dict = {}
select_type = ["Sequence", "Prediction", "Interaction",]

prompt_template_input = {
    "Sequence": {
        "T1": {
            "question": r"^Which object did the person ([\w\s'/]+) after they ([\w\s'/]+) the ([\w\s'/]+)\?",
            "answer": r"^The ([\w\s'/]+)\.",
            "type": ["verb_2", "verb_1", "noun_1", "noun_2",],
        },
        "T2": {
            "question": r"^Which object did the person ([\w\s'/]+) before they ([\w\s'/]+) the ([\w\s'/]+)\?",
            "answer": r"^The ([\w\s'/]+)\.",
            "type": ["verb_1", "verb_2", "noun_2", "noun_1",],
        },
        "T3": {
            "question": r"^What happened after the person ([\w\s'/]+) the ([\w\s'/]+)\?",
            "answer": r"^([\w\s'/]+) the ([\w\s'/]+)\.",
            "type": ["verb_1", "noun_1", "verb_2", "noun_2",],
        },
        "T4": {
            "question": r"^What happened before the person ([\w\s'/]+) the ([\w\s'/]+)\?",
            "answer": r"^([\w\s'/]+) the ([\w\s'/]+)\.",
            "type": ["verb_2", "noun_2", "verb_1", "noun_1",],
        },
        "T5": {
            "question": r"^What did the person do to the ([\w\s'/]+) after ([\w\s'/]+) the ([\w\s'/]+)\?",
            "answer": r"^([\w\s'/]+)\.",
            "type": ["noun_2", "verb_1", "noun_1", "verb_2",],
        },
        "T6": {
            "question": r"^What did the person do to the ([\w\s'/]+) before ([\w\s'/]+) the ([\w\s'/]+)\?",
            "answer": r"^([\w\s'/]+)\.",
            "type": ["noun_1", "verb_2", "noun_2", "verb_1",],
        },
    },
    "Prediction": {
        "T1": {
            "question": r"^What will the person do next?",
            "answer": r"^([\w\s'/]+) the ([\w\s'/]+)\.",
            "type": ["verb_1", "noun_1",],
        },
        "T2": {
            "question": r"^What will the person do next with the ([\w\s'/]+)\?",
            "answer": r"^([\w\s'/]+)\.",
            "type": ["noun_1", "verb_1",],

        },
        "T3": {
            "question": r"^Which object would the person ([\w\s'/]+) next\?",
            "answer": r"^The ([\w\s'/]+)\.",
            "type": ["verb_1", "noun_1",],

        },
        "T4": {
            "question": r"^Which object would the person ([\w\s'/]+) next after they ([\w\s'/]+) the ([\w\s'/]+)\?",
            "answer": r"^The ([\w\s'/]+)\.",
            "type": ["verb_2", "verb_1", "noun_1", "noun_2", ],

        },
    }
}

prompt_target = {
    "Sequence": [
        [
            {
                "question": "During {start_time} seconds to {end_time} seconds, the person did two things. Please list them sequentially.",
                "question_state": ["start_time", "end_time",],
                "answer": ["{verb_1_ing} the {noun_1}.", "{verb_2_ing} the {noun_2}.",],
                "answer_state": ["verb_1_ing", "noun_1", "verb_2_ing", "noun_2",],
                "answer_type": "verb+noun",
                "add": "Answer the above question according to the video. Only use words from the following words to organize your answer.",
                "type": "S00",
            },
            {
                "question": "What did the person do from {start_time} seconds to {end_time} seconds? List the things they do sequentially.",
                "question_state": ["start_time", "end_time",],
                "answer": ["{verb_1_ing} the {noun_1}.", "{verb_2_ing} the {noun_2}.",],
                "answer_state": ["verb_1_ing", "noun_1", "verb_2_ing", "noun_2",],
                "answer_type": "verb+noun",
                "add": "Answer the above question according to the video. Only use words from the following words to organize your answer.",
                "type": "S01",
            },
            {
                "question": "The person did A {loca} B between {start_time} seconds and {end_time} seconds. What are A and B?",
                "question_state": ["loca", "start_time", "end_time",],
                "answer": ["{verb_1_ing} the {noun_1}.", "{verb_2_ing} the {noun_2}.",],
                "answer_state": ["verb_1_ing", "noun_1", "verb_2_ing", "noun_2",],
                "answer_type": "verb+noun",
                "add": "Answer the above question according to the video. Only use words from the following words to organize your answer.",
                "type": "S02",
            },
            {
                "question": "Focus on the segment {start_time} seconds - {end_time} seconds. What did the person do {loca} they {verb_2_ed} the {noun_2}?",
                "question_state": ["start_time", "end_time", "loca", "verb_2_ed", "noun_2",],
                "answer": ["{verb_1_ed} the {noun_1}.",],
                "answer_state": ["verb_1_ed", "noun_1",],
                "answer_type": "verb+noun",
                "add": "Answer the above question according to the video. Only use words from the following words to organize your answer.",
                "type": "S03",
            },
        ],
        [
            {
                "question": "Which object did the person {verb_1_base} {loca} they {verb_2_ed} ___? ___.",
                "question_state": ["verb_1_base", "loca", "verb_2_ed", "start_time", "end_time",],
                "answer": ["The {noun_2}.", "The {noun_1}.",],
                "answer_state": ["noun_2", "noun_1",],
                "answer_type": "noun",
                "add": "Choose words from the following words to fill in the blanks according to segment {start_time} seconds - {end_time} seconds of the video.",
                "type": "S10",
            },
            {
                "question": "The person ___ {loca} they ___.",
                "question_state": ["loca", "start_time", "end_time",],
                "answer": ["{verb_1_ed} the {noun_1}.", "{verb_2_ed} the {noun_2}.",],
                "answer_state": ["verb_1_ed", "noun_1", "verb_2_ed", "noun_2",],
                "answer_type": "verb+noun",
                "add": "Choose words from the following words to fill in the blanks according to segment {start_time} seconds - {end_time} seconds of the video.",
                "type": "S11",
            },
        ]
    ],
    "Prediction": [
        {
            "question": "What will the person do after {start_time} seconds - {end_time} seconds?",
            "question_state": ["start_time", "end_time",],
            "answer": ["{verb_1_base} the {noun_1}.",],
            "answer_state": ["verb_1_base", "noun_1",],
            "answer_type": "verb+noun",
            "add": "Only use words from the following words to organize your answer.",
            "type": "P0",
        },
        {
            "question": "What will the person do next with the {noun_1} after {start_time} seconds - {end_time} seconds?",
            "question_state": ["noun_1", "start_time", "end_time",],
            "answer": ["{verb_1_base}.",],
            "answer_state": ["verb_1_base",],
            "answer_type": "verb",
            "add": "Choose answer from the following options.",
            "type": "P1",
        },
        {
            "question": "Which object would the person {verb_1_base} after {start_time} seconds - {end_time} seconds?",
            "question_state": ["verb_1_base", "start_time", "end_time",],
            "answer": ["{noun_1}.",],
            "answer_state": ["noun_1",],
            "answer_type": "noun",
            "add": "Choose answer from the following options.",
            "type": "P2",
        },
        {
            "question": "According to {start_time} seconds - {end_time} seconds, which object would the person {verb_2_base} next after they {verb_1_base} the {noun_1}?",
            "question_state": ["start_time", "end_time", "verb_1_base", "noun_1", "verb_2_base",],
            "answer": ["{noun_2}.",],
            "answer_state": ["noun_2",],
            "answer_type": "noun",
            "add": "Choose answer from the following options.",
            "type": "P3",
        },
    ],
    "Interaction": {
        "question_add": "Choose answer from the following options according to fragment {start_time} seconds - {end_time} seconds of the video.",
        "answer_type": "verb+noun",
        "type": "I0",
    },
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

def generate_question_answer_templates_Interaction(rec: dict):
    template_id = rec["question_id"].split('_')

    prompt_target_choice = prompt_target["Interaction"]

    fixed_question = " ".join(
        [rec["question"], prompt_target_choice["question_add"]]
    ).format(
        start_time = rec["start"],
        end_time = rec["end"],
    )

    fixed_answer = rec["answer"]

    fixed_options = [item["choice"] for item in rec["choices"]]

    prompt_type = "->".join(["_".join(template_id[:-1]), prompt_target_choice["type"],])

    return fixed_question, fixed_answer, fixed_options, prompt_type, None

def generate_question_answer_templates(rec: dict):
    template_id = rec["question_id"].split('_')

    if template_id[0] == "Interaction":
        return generate_question_answer_templates_Interaction(rec)

    question_template = prompt_template_input[template_id[0]][template_id[1]]['question']
    answer_template = prompt_template_input[template_id[0]][template_id[1]]['answer']
    re_type = prompt_template_input[template_id[0]][template_id[1]]['type']

    question_time_template = 'before' if 'before' in rec["question"] else 'after'
    question_match = re.search(question_template, rec["question"])
    answer_match = re.search(answer_template, rec["answer"])

    if not question_match or not answer_match:
        raise ValueError(
            f"Template '{question_match} or {answer_match}' not found in question or answer: '{rec["question"]}' or '{rec["answer"]}'"
        )

    question_match_group = question_match.groups() if question_match.lastindex else ()
    answer_match_group = answer_match.groups() if answer_match.lastindex else ()
    mixed_match_group = question_match_group + answer_match_group

    prompt_target_choice = random.choice(prompt_target[template_id[0]])
    if template_id[0] == "Sequence":
        prompt_target_choice = random.choice(prompt_target_choice)
    elif template_id[0] == "Prediction":
        if len(answer_match_group) == 4:
            prompt_target_choice = prompt_target[template_id[0]][-1]
        else:
            prompt_target_choice = random.choice(prompt_target[template_id[0]][:-1])
    else:
        raise ValueError()

    dict_IR = {}
    IR_question = {}
    IR_answer = {}

    if len(re_type) != len(mixed_match_group):
        raise ValueError()

    if "Sequence_T1" in rec["question_id"]:
        pass

    for type, match in zip(re_type, mixed_match_group):
        dict_IR.update(
            {
                type: match
            }
        )
    middle_state = dict_IR.copy()

    IR_list_rec = ["start_time", "end_time",]
    IR_list_loc = ["loca",]
    IR_list_noun = ["noun_1", "noun_2",]
    IR_list_verb = [
        "verb_1_base", "verb_2_base",
        "verb_1_ing", "verb_2_ing",
        "verb_1_ed", "verb_2_ed",
    ]

    # switch before and after
    if len(dict_IR) == 4 and IR_list_loc[0] in prompt_target_choice["question"] and question_time_template == 'after':
        temp_ = dict_IR['verb_1']
        dict_IR['verb_1'] = dict_IR['verb_2']
        dict_IR['verb_2'] = temp_
        temp_ = dict_IR['noun_1']
        dict_IR['noun_1'] = dict_IR['noun_2']
        dict_IR['noun_2'] = temp_



    if debug:
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        print(dict_IR)
        print(IR_question)
        print(IR_answer)
        print(question_template)
        print(prompt_target_choice["question"])
        print(prompt_target_choice["answer"])
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")

    question_template_fixed = " ".join([prompt_target_choice["question"], prompt_target_choice["add"]])
    for IR_item in prompt_target_choice["question_state"]:
        if IR_item in IR_list_rec:
            IR_question.update({IR_item: rec[IR_item.split('_')[0]], }, )
        if IR_item in IR_list_loc:
            IR_question.update({IR_item: question_time_template, }, )
        if IR_item in IR_list_noun:
            IR_question.update({IR_item: dict_IR[IR_item], }, )
        for IR_item_ in IR_list_verb:
            if IR_item in IR_item_:
                vber_change = get_verb_forms(dict_IR["_".join(IR_item.split('_')[:-1])])
                if "base" in IR_item_:
                    IR_question.update({IR_item_: vber_change["base"], }, )
                elif "ed" in IR_item_:
                    IR_question.update({IR_item_: vber_change["past"], }, )
                elif "ing" in IR_item_:
                    IR_question.update({IR_item_: vber_change["present_participle"], }, )
                else:
                    raise ValueError()

    fixed_question = question_template_fixed.format(**IR_question)

    fixed_options = []

    option_state = "base"
    for IR_item in prompt_target_choice["answer_state"]:
        if IR_item in IR_list_rec:
            IR_answer.update({IR_item: rec[IR_item], }, )
        if IR_item in IR_list_loc:
            IR_answer.update({IR_item: question_time_template, }, )
        if IR_item in IR_list_noun:
            IR_answer.update({IR_item: dict_IR[IR_item], }, )
            fixed_options.append(" ".join(["The", dict_IR[IR_item]]))
        for IR_item_ in IR_list_verb:
            if IR_item in IR_item_:
                vber_change = get_verb_forms(dict_IR["_".join(IR_item.split('_')[:-1])])
                if "base" in IR_item_:
                    IR_answer.update({IR_item_: vber_change["base"], }, )
                    option_state = "base"
                elif "ed" in IR_item_:
                    IR_answer.update({IR_item_: vber_change["past"], }, )
                    option_state = "past"
                elif "ing" in IR_item_:
                    IR_answer.update({IR_item_: vber_change["present_participle"], }, )
                    option_state = "present_participle"
                else:
                    raise ValueError()
                fixed_options.append(vber_change[option_state])

    if debug:
        print("BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB")
        print(dict_IR)
        print(IR_question)
        print(IR_answer)
        print(prompt_target_choice["question"])
        print(prompt_target_choice["answer"])
        print("BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB")


    fixed_answer = [_.format(**IR_answer) for _ in prompt_target_choice["answer"]]

    IR_list_option = ["verb", "noun",]
    for IR_item in prompt_target_choice["answer_type"]:
        if "noun" in IR_item:
            fixed_options.extend(random.choices(select_options[IR_item], k=random.randint(4, 6)))
        if "verb" in IR_item:
            temp_verb = [get_verb_forms(_)[option_state] for _ in random.choices(select_options[IR_item], k=random.randint(4, 6))]
            fixed_options.extend(temp_verb)

    fixed_options = [_ + '.' for _ in list(set(fixed_options))]

    prompt_type = "->".join(["_".join(template_id[:-1]), prompt_target_choice["type"],])

    for loc in IR_list_loc:
        if loc in prompt_target_choice["question_state"]:
            prompt_type = "_".join([prompt_type, IR_question[loc],])

    return fixed_question, fixed_answer, fixed_options, prompt_type, middle_state

def json_file_creator(rec: dict) -> dict:

    fixed_question, fixed_answer, fixed_options, prompt_type, middle_state = generate_question_answer_templates(rec)

    fixed_json_file = \
        {
            "dataset": json_label,
            "task": rec['question_id'].split('_')[0],
            "fixed_type": prompt_type,
            'id': rec['question_id'],
            "video": rec['video_id'] + '.mp4',
            'tgt': [
                rec['start'],
                rec['end'],
            ],
            "original_question": rec['question'],
            "original_answer": rec['answer'],
            "middle_state": middle_state,
            "conversations": [
                {
                    "from": "human",
                    "value": fixed_question,
                },
                {
                    "from": "gpt",
                    "type": "select_option",
                    "value": fixed_answer,
                },
            ],
            "options": fixed_options,
        }

    return fixed_json_file

def main():
    # read files
    for json_file_name in tqdm.tqdm(json_file_list, total=len(json_file_list), desc="Processing json files"):
        with open(os.path.join(json_path, json_file_name), 'r') as file:
            json_files_dict[json_file_name] = json.load(file)

    for json_file_name, json_data in tqdm.tqdm(json_files_dict.items(), total=len(json_files_dict), desc="Processing json files"):
        for rec in tqdm.tqdm(json_data, total=len(json_data), desc="Processing json files"):
            for _ in rec["choices"]:
                select_options['noun' if _["choice"].startswith("The ") else 'verb'].append(_["choice"])

    for index, json_list in tqdm.tqdm(json_files_dict.items(), total=len(json_files_dict), desc="Processing json files"):
        json_files_dict[index] = [
            rec for rec in json_list
            if rec['question_id'].split('_')[0] in select_type
        ]

    fixed_json_files_dict = {}
    for index, json_list in json_files_dict.items():
        fixed_json_list = []
        for _ in json_list:
            fixed_json_list.append(json_file_creator(_))
        # with multiprocessing.Pool(processes = 16) as pool:
        #     fixed_json_list = list(tqdm.tqdm(pool.imap(json_file_creator, json_list),total=len(json_list), desc="Processing json files"))


        # fixed_json_list = [_ for _ in fixed_json_list if _ is not None]

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