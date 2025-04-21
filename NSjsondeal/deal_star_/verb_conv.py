from lemminflect import getLemma, getInflection

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