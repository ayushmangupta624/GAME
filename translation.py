from config import model, SAFETY_SETTINGS, GENERATION_CONFIG


def seg_trans(seg):
    response = model.generate_content(
        f"Translate the following English segment into Hindi: {seg}. "
        f"The output should only contain translated segment and no explanation",
        safety_settings=SAFETY_SETTINGS,
        generation_config=GENERATION_CONFIG
    )
    return response.text


def seg2_trans(seg):
    response = model.generate_content(
        f"Translate the following Hindi segment into English: {seg}. "
        f"The output should only contain translated segment and no explanation",
        safety_settings=SAFETY_SETTINGS,
        generation_config=GENERATION_CONFIG
    )
    return response.text


def w_trans_pos(word, pos, en_sen):
    response = model.generate_content(
        f"Translate the following English word into Hindi: {word}. "
        f"The POS tag of the word is {pos}. "
        f"Infer the context from the English sentence: {en_sen}. "
        f"For past tense verbs, add a 'kiya' if necessary. "
        f"Give the output without any explanation",
        safety_settings=SAFETY_SETTINGS,
        generation_config=GENERATION_CONFIG
    )
    return response.text


def sen_trans(sen):
    response = model.generate_content(
        f"Given a sentence transliterated in Hindi, your task is to translate it to English. "
        f"Sentence: {sen}. The output should only contain transliteration and no explanation",
        safety_settings=SAFETY_SETTINGS,
        generation_config=GENERATION_CONFIG
    )
    return response.text


def e2h(sen):
    response = model.generate_content(
        f"Given an English sentence, translate it to Hindi. "
        f"English sentence: {sen}. "
        f"The output should only contain translated segment and no explanation",
        safety_settings=SAFETY_SETTINGS,
        generation_config=GENERATION_CONFIG
    )
    return response.text


def aop(sen):
    response = model.generate_content(
        f"Given a Hindi sentence, determine whether the verb is active or passive. "
        f"If the verb is active, output '1'. If the verb is passive, output '0'. "
        f"Hindi sentence: {sen}",
        safety_settings=SAFETY_SETTINGS,
        generation_config=GENERATION_CONFIG
    )
    return response.text