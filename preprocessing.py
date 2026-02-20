import string
import nltk
from nltk import word_tokenize

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')


def preprocess(sen):
  sen = sen.lower()
  tr = str.maketrans('', '', string.punctuation)
  sen = sen.translate(tr)
  sen = ' '.join(sen.split())

  return sen

def assign_tags(s_cm, lid_res):
    def reconstruct_word(tokens):
        return ''.join([token['word'].replace('##', '') for token in tokens])

    # Sort lid_res by start index
    lid_res = sorted(lid_res, key=lambda x: x['start'])

    word_tags = {}
    current_word_tokens = []

    for i, token in enumerate(lid_res):
        if i == 0:
            current_word_tokens.append(token)
        else:
            prev_token = lid_res[i - 1]
            if prev_token['end'] == token['start']:
                # Combine current token with the previous one
                current_word_tokens.append(token)
            else:
                # End of the current word, process and start a new one
                word_str = reconstruct_word(current_word_tokens)
                word_tags[word_str] = max(current_word_tokens, key=lambda x: x['score'])['entity']
                current_word_tokens = [token]

    # Handle the last word
    if current_word_tokens:
        word_str = reconstruct_word(current_word_tokens)
        word_tags[word_str] = max(current_word_tokens, key=lambda x: x['score'])['entity']

    # Split the sentence into words and assign tags
    words = s_cm.split()
    wdl = [{'word': word, 'ltag': '', 'pos': ''} for word in words]

    for i, word in enumerate(words):
        if word in word_tags:
            wdl[i]['ltag'] = word_tags[word]
        else:
            wdl[i]['ltag'] = 'unknown'

    return wdl
