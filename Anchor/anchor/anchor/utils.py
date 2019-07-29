"""bla"""
# from __future__ import print_function
import copy
import sklearn
import numpy as np
import lime
import os
import sys

if (sys.version_info > (3, 0)):
    def unicode(s, errors=None):
        return s#str(s)


class Neighbors:
    def __init__(self, nlp_obj):
        self.nlp = nlp_obj
        self.to_check = [w for w in self.nlp.vocab if w.prob >= -15]
        self.n = {}

    def neighbors(self, word):
        word = unicode(word)
        orig_word = word
        if word not in self.n:
            if word not in self.nlp.vocab:
                self.n[word] = []
            else:
                word = self.nlp.vocab[word]
                queries = [w for w in self.to_check
                            if w.is_lower == word.is_lower]
                if word.prob < -15:
                    queries += [word]
                by_similarity = sorted(
                    queries, key=lambda w: word.similarity(w), reverse=True)
                self.n[orig_word] = [(self.nlp(w.orth_)[0], word.similarity(w))
                                     for w in by_similarity[:500]]
        return self.n[orig_word]

def perturb_sentence(text, present, n, neighbors, proba_change=0.5,
                     top_n=50, forbidden=[], forbidden_tags=['PRP$'],
                     forbidden_words=['be'],
                     pos=['NOUN', 'VERB', 'ADJ', 'ADV', 'ADP', 'DET'], use_proba=True,
                     temperature=.4):
    # words is a list of words (must be unicode)
    # present is which ones must be present, also a list
    # n = how many to sample
    # neighbors must be of utils.Neighbors
    # nlp must be spacy
    # proba_change is the probability of each word being different than before
    # forbidden: forbidden lemmas
    # forbidden_tags, words: self explanatory
    # pos: which POS to change

    tokens = neighbors.nlp(unicode(text))
    eligible = []
    forbidden = set(forbidden)
    forbidden_tags = set(forbidden_tags)
    forbidden_words = set(forbidden_words)
    pos = set(pos)
    raw = np.zeros((n, len(tokens)), '|S80')
    data = np.ones((n, len(tokens)))
    raw[:] = [x.text for x in tokens]
    for i, t in enumerate(tokens):
        if i in present:
            continue
        if (t.text not in forbidden_words and t.pos_ in pos and
                t.lemma_ not in forbidden and t.tag_ not in forbidden_tags):
            r_neighbors = [
                (unicode(x[0].text.encode('utf-8'), errors='ignore'), x[1])
                for x in neighbors.neighbors(t.text)
                if x[0].tag_ == t.tag_][:top_n]
            if not r_neighbors:
                continue
            t_neighbors = [x[0] for x in r_neighbors]
            weights = np.array([x[1] for x in r_neighbors])
            if use_proba:
                weights = weights ** (1. / temperature)
                weights = weights / sum(weights)
                # print t.text
                # print sorted(zip(t_neighbors, weights), key=lambda x:x[1], reverse=True)[:10]
                raw[:, i] = np.random.choice(t_neighbors, n,  p=weights,
                                             replace=True)
                # The type of data in raw is byte.
                data[:, i] = raw[:, i] == t.text.encode()
            else:
                n_changed = np.random.binomial(n, proba_change)
                changed = np.random.choice(n, n_changed, replace=False)
                if t.text in t_neighbors:
                    idx = t_neighbors.index(t.text)
                    weights[idx] = 0
                weights = weights / sum(weights)
                raw[changed, i] = np.random.choice(t_neighbors, n_changed, p=weights)
                data[changed, i] = 0
#         else:
#             print t.text, t.pos_ in pos, t.lemma_ in forbidden, t.tag_ in forbidden_tags, t.text in neighbors
    # print raw
    if (sys.version_info > (3, 0)):
        raw = [' '.join([y.decode() for y in x]) for x in raw]
    else:
        raw = [' '.join(x) for x in raw]
    return raw, data
