import os
import re
import sys
import random
import json
from nltk.corpus import wordnet
import multiprocess
import tqdm

json_path = "/Users/lsf/PycharmProjects/DatasetJson/NSjsondeal/Complise/star"
save_path = "/Users/lsf/PycharmProjects/DatasetJson/NSjsondeal/Complise/star/save"
json_file_list = os.listdir(json_path)
json_file_list = [rec for rec in json_file_list if rec.endswith(".json") and not rec.endswith("fix.json")]
json_label = json_path.split('/')[-1]

select_type = ["Sequence", "Prediction", ]


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
    # v base v
    # d past v
    # n n
    #k keep
    "Sequence": [
        [
            ("During {} to {}, the person did two things. Please list them sequentially.", 'se',),
            ("What did the person do from {} to {}? List the things they do sequentially.", 'se',),
            ("The person did A {} B between {} and {}. What are A and B?", 'bse',),
            ("Focus on the segment {} - {}. What did the person do {} they {}?", 'sebo',),
            ("Answer the above question according to the video. Only use words from the following words to organize your answer.", '',),
        ],
        # [
        #     ("Which object did the person {} {} they {} the ___ ? The ___.", 'vbd',),
        #     ("The person ___ the ___ {} they ___ the ___.", 'b',),
        #     ("Choose words from the following words to fill in the blanks according to segment {} - {} of the video.", 'se',),
        # ]
    ],
    "Prediction":[
        ("What will the person do after {} - {}?", "se",),
        ("What will the person do next with the {} after {} - {}?", "nse",),
        ("Which object would the person {} after {} - {}?", "vse",),
        ("According to {} - {}, which object would the person {} next after they {} the {}?", "sekkk",),
        ("Choose answer from the following options.", "",)
    ]

}

def regular_past_tense(verb):
    """规则化生成过去式"""
    if verb.endswith('e'):
        return verb + 'd'
    elif verb.endswith('y') and len(verb) > 1 and verb[-2] not in 'aeiou':
        return verb[:-1] + 'ied'
    elif verb.endswith(('c', 'g')) and len(verb) > 1 and verb[-2] in 'aeiou':
        return verb + 'ked' if verb.endswith('c') else verb + 'ged'
    elif len(verb) > 1 and verb[-1] in 'bcdfghjklmnpqrstvwxz' and verb[-2] in 'aeiou':
        if len(verb) == 2 or (len(verb) > 2 and verb[-3] not in 'aeiou'):
            return verb + verb[-1] + 'ed'
    return verb + 'ed'

def present_participle(verb):
    """规则化生成现在分词"""
    if verb.endswith('e') and len(verb) > 1:
        if verb.endswith('ie'):
            return verb[:-2] + 'ying'
        elif verb not in ('be', 'see', 'flee', 'knee'):
            return verb[:-1] + 'ing'
    elif verb.endswith('y') and len(verb) > 1 and verb[-2] not in 'aeiou':
        return verb + 'ing'
    elif len(verb) > 2 and verb[-1] in 'bcdfghjklmnpqrstvwxz' and verb[-2] in 'aeiou' and verb[-3] not in 'aeiou':
        return verb + verb[-1] + 'ing'
    return verb + 'ing'

def get_verb_forms(word):
    """
    获取动词的三种形式：原型(base)、过去式(past)、现在分词(present participle/ing)
    返回一个字典：{'base': ..., 'past': ..., 'ing': ...}
    """
    forms = {'base': word, 'past': None, 'ing': None}

    all_words = word.split()
    word = all_words[0]

    # 首先尝试从WordNet获取信息
    synsets = wordnet.synsets(word, pos=wordnet.VERB)
    if synsets:
        lemmas = synsets[0].lemmas()
        if lemmas:
            related_forms = lemmas[0].derivationally_related_forms()

            # 查找过去式
            for form in related_forms:
                if form.relationship().name() == 'past_tense':
                    forms['past'] = form.name()
                    break

            # 查找现在分词
            for form in related_forms:
                if form.relationship().name() == 'present_participle':
                    forms['ing'] = form.name()
                    break

    # 如果WordNet中没有找到过去式，使用规则生成
    if forms['past'] is None:
        forms['past'] = regular_past_tense(word)

    # 如果WordNet中没有找到现在分词，使用规则生成
    if forms['ing'] is None:
        forms['ing'] = present_participle(word)


    forms['ing'] = "".join([forms['ing'], all_words[1:]])
    forms['base'] = "".join([forms['base'], all_words[1:]])
    forms['past'] = "".join([forms['past'], all_words[1:]])
    return forms

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
            func_list.append(matched_groups[ind])
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

    fixed_question = fixed_question.join([choice_prompt])

    fixed_options = [item["choice"] for item in rec["choices"]]

    for index, item in enumerate(answers_list):
        if item[-1] != '.':
            answers_list[index] = item + '.'

    fixed_options.extend(answers_list)
    fixed_options = list(set(fixed_options))
    random.shuffle(fixed_options)

    return fixed_question, fixed_options, answers_list

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

    create_option = "Q+A" if len(matched_groups) > 2 else ""

    answers_list = []

    # if create_option == "Q+A":
    #     answers_list.append((get_verb_forms(matched_groups[0])['ing'] + " " + rec["answer"]).lower())
    #     answers_list.append((get_verb_forms(matched_groups[1])['ing'] + " " + matched_groups[2]).lower())
    # else:
    #     answers_list.append((get_verb_forms(matched_groups[0])['ing'] + " " + matched_groups[1]).lower())
    #     answers_list.append((get_verb_forms(rec["answer"][:-1])['ing'] + " " + matched_groups[2]).lower())

    if create_option == "Q+A":
        answers_list.append((matched_groups[0] + " the " + rec["answer"]).lower())
        answers_list.append((matched_groups[1] + " the " + matched_groups[2]).lower())
    else:
        answers_list.append((matched_groups[0] + " the " + matched_groups[1]).lower())
        # temp_answer_tamplate = rec["answer"].split(" the ")
        # answers_list.append((temp_answer_tamplate[0] + " the " + temp_answer_tamplate[2]).lower())
        answers_list.append(rec["answer"].lower())


    fixed_question = ""

    answers_list = [_.capitalize() for _ in answers_list]

    prompt_target_choice = random.choice(prompt_target[target_id[0]])

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
            func_list.append(answers_list[0])
            answers_list = [answers_list[1]]
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

    fixed_question = fixed_question.join([choice_prompt])

    fixed_options = [item["choice"] for item in rec["choices"]]

    for index, item in enumerate(answers_list):
        if item[-1] != '.':
            answers_list[index] = item + '.'

    fixed_options.extend(answers_list)
    fixed_options = list(set(fixed_options))
    random.shuffle(fixed_options)

    return fixed_question, fixed_options, answers_list



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
            if rec['question_id'].split('_')[0] in select_type
        ]

    # change dataset
    fixed_json_files_dict = {}
    for index, json_list in json_files_dict.items():
        fixed_json_list = []
        for rec in tqdm.tqdm(json_list, total=len(json_list), desc=f"Processing {index}"):
            # try:
                fixed_question, fixed_options, fixed_answers = [], [], []
                if rec['question_id'].split('_')[0] == "Sequence":
                    fixed_question, fixed_options, fixed_answers = generate_question_templates_Sequence(rec)
                elif rec['question_id'].split('_')[0] == "Prediction":
                    fixed_question, fixed_options, fixed_answers = generate_question_templates_Prediction(rec)

                fixed_json_list.append(
                    {
                        "dataset": json_label,
                        "task": rec['question_id'].split('_')[0],
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