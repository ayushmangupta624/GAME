import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import torch
from sentence_transformers import util
from google.transliteration import transliterate_text
from codeswitch.codeswitch import LanguageIdentification

from config import sim_model, TRANSLIT_LOOKUP, EXCEPTIONS
from preprocessing import preprocess, assign_tags
from translation import seg_trans, w_trans_pos, sen_trans, e2h, aop

lemmatizer = WordNetLemmatizer()
lid = LanguageIdentification('hin-eng')


def json_gen(en_sen):
    json_dict = []
    pos_tags = nltk.pos_tag(word_tokenize(en_sen))
    hin_sen = e2h(en_sen)
    voice = aop(hin_sen)
    hin_verb_type = "ACTIVE" if voice != 0 else "PASSIVE"

    for word, pos in pos_tags:
        if pos in ['VBN', 'VBP', 'VBZ', 'VBG', 'VBD', 'VB', 'VERB', 'Verb']:
            hi_word = w_trans_pos(word, pos, en_sen)
            hi = nltk.word_tokenize(hi_word)
            if hi[-1] == "है" and len(hi) != 1:
                hi_word = hi_word[:-3]
            if hi[-1] == 'किया':
                base_hin = hi_word[:-4] + "कर" + "ना"
            else:
                base_hin = hi_word[:-2] + "ना"
            base_eng = lemmatizer.lemmatize(word, pos="v")
            json_dict.append({
                'eng': word,
                'pos_tag': pos,
                'hindi': hi_word,
                'base_hin': base_hin,
                'base_eng': base_eng,
                'hin_verb_type': hin_verb_type
            })
    return json_dict, hin_sen


def inflection_cm_gen(en, cm):
    json_dict, hin_sen = json_gen(en)
    max_possible_errors = len(json_dict)
    hin_sen = preprocess(hin_sen)
    hindi_sentence_list = nltk.word_tokenize(hin_sen)
    hindi_sentence_list = [word.rstrip('।') for word in hindi_sentence_list]
    hindi_sentence_list2 = hindi_sentence_list.copy()
    rep_suffix = ''

    for i in range(len(json_dict)):
        temp_dict_1 = json_dict[i]
        temp_dict_1["hindi"] = preprocess(temp_dict_1["hindi"])
        if temp_dict_1['pos_tag'] in ['VBN', 'VBP', 'VBZ', 'VBG', 'VBD', 'VB', 'VERB', 'Verb']:
            index_1stword = hindi_sentence_list.index((temp_dict_1["hindi"].split())[0])
            temp_base_word = (temp_dict_1["base_hin"].split())[0]
            if (temp_base_word[-2] + temp_base_word[-1]) == "ना":
                rema_base_word = temp_base_word[0:-2]
                base_in_hin = temp_dict_1["hindi"][0:len(rema_base_word)]
                rem_in_hin = (temp_dict_1["hindi"]).split()[0][len(rema_base_word):]
            else:
                rem_in_hin = 'NA'
                rep_suffix = ''
            if rem_in_hin == "ना":
                if hindi_sentence_list[index_1stword + 1] in ['|', 'और', '।'] or index_1stword == len(hin_sen) - 1:
                    rep_suffix = 'करना'
                else:
                    rep_suffix = ''
            elif rem_in_hin in ["ने", "नी", "ता", "ती", "ते", "ो"]:
                if temp_dict_1['hin_verb_type'] == 'PASSIVE':
                    rep_suffix = 'हो' + rem_in_hin
                else:
                    rep_suffix = 'कर' + rem_in_hin
            elif rem_in_hin == '':
                if hindi_sentence_list[index_1stword + 1] in ['करता', 'करती', 'करते', 'कर', 'करें', 'करो', 'हुआ', 'हुए', 'हुई']:
                    rep_suffix = ''
                else:
                    rep_suffix = "कर"
            elif rem_in_hin in ['ा', 'े', 'या']:
                if hindi_sentence_list[index_1stword + 1] in ['हुआ', 'हुए']:
                    rep_suffix = ''
                elif temp_dict_1['hin_verb_type'] == 'PASSIVE':
                    if hindi_sentence_list[index_1stword + 1] in ['गया', 'गई', 'गए']:
                        rep_suffix = 'किय' + rem_in_hin
                    else:
                        rep_suffix = 'hu' + rem_in_hin
                else:
                    rep_suffix = 'किय' + rem_in_hin
            elif rem_in_hin in ['एं', 'ें']:
                rep_suffix = 'करें'
            elif rem_in_hin in ['ी', 'ई']:
                if hindi_sentence_list[index_1stword + 1] in ['हुई', 'की']:
                    rep_suffix = ''
                elif temp_dict_1['hin_verb_type'] == 'PASSIVE':
                    rep_suffix = 'हुई'
                else:
                    rep_suffix = 'की'

    hindi_sentence_list2[index_1stword] = (temp_dict_1["base_eng"] + ' ' + rep_suffix).strip()
    mod_sen = ' '.join(hindi_sentence_list2)
    mod_sen_list = mod_sen.split()
    transliterated_words = [
        TRANSLIT_LOOKUP[word] if word in TRANSLIT_LOOKUP else word for word in mod_sen_list
    ]
    verb_pref = f"{transliterated_words[index_1stword]} {transliterated_words[index_1stword + 1]}"
    k = 0
    if verb_pref not in cm:
        k += 1
    return k, max_possible_errors


def evaluate(s_r, s_cm):
    s_r = preprocess(s_r)
    s_cm = preprocess(s_cm)

    lid_res = lid.identify(s_cm)
    wdl = assign_tags(s_cm, lid_res)

    seq = []
    cseq = []
    for wd in wdl:
        if wd['ltag'] == 'en' and wd['word'] not in EXCEPTIONS:
            cseq.append(wd['word'])
        elif cseq:
            seq.append(" ".join(cseq))
            cseq = []
    if cseq:
        seq.append(" ".join(cseq))

    for seg in seq:
        tseg = seg_trans(seg)
        tseg = tseg.replace("।", "")
        s_cm = s_cm.replace(seg, tseg)

    s_cm = transliterate_text(s_cm, lang_code='hi')

    s_en = sen_trans(s_cm)
    s_en = preprocess(s_en)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    e_r = sim_model.encode(s_r, convert_to_tensor=True).to(device)
    e_en = sim_model.encode(s_en, convert_to_tensor=True).to(device)

    sim = util.pytorch_cos_sim(e_r, e_en).item()
    return sim