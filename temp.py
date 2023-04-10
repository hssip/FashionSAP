import torch

from nltk.corpus import wordnet as wn
import random

# filename = '/mnt/MDisk/hyp/out_albef/fashiongen_pretrain/checkpoint_best.pth'

# state = torch.load(filename, map_location='cpu')
# print(state['model'].keys())

def search_antonym(word, label=None):
    anto = []
    for syn in wn.synsets(word):
        for lm in syn.lemmas():
            if lm.antonyms():
                anto.append(lm.antonyms()[0].name())
    return random.choice(anto) if anto else word

print(search_antonym('one'))


