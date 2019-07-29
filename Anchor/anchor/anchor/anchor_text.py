from . import utils
from . import anchor_base
from . import anchor_explanation
import numpy as np
import json
import os
import string
import sys
from io import open

# Python3 hack
try:
    UNICODE_EXISTS = bool(type(unicode))
except NameError:
    def unicode(s):
        return s


def id_generator(size=15):
    """Helper function to generate random div ids. This is useful for embedding
    HTML into ipython notebooks."""
    chars = list(string.ascii_uppercase + string.digits)
    return ''.join(np.random.choice(chars, size, replace=True))


class AnchorText(object):
    """bla"""
    def __init__(self, nlp, class_names, use_unk_distribution=True):
        """
        Args:
            nlp: spacy object
            class_names: list of strings
            use_unk_distribution: if True, the perturbation distribution
                will just replace words randomly with UNKs.
                If False, words will be replaced by similar words using word
                embeddings
        """
        self.nlp = nlp
        self.class_names = class_names
        self.neighbors = utils.Neighbors(self.nlp)
        self.use_unk_distribution = use_unk_distribution

    def get_sample_fn(self, text, classifier_fn, use_proba=False):
        true_label = classifier_fn([text])[0]
        processed = self.nlp(unicode(text))
        words = [x.text for x in processed]
        positions = [x.idx for x in processed]

        def sample_fn(present, num_samples, compute_labels=True):
            if self.use_unk_distribution:
                data = np.ones((num_samples, len(words)))
                raw = np.zeros((num_samples, len(words)), '|S80')
                raw[:] = words
                for i, t in enumerate(words):
                    if i in present:
                        continue
                    n_changed = np.random.binomial(num_samples, .5)
                    changed = np.random.choice(num_samples, n_changed,
                                               replace=False)
                    raw[changed, i] = 'UNK'
                    data[changed, i] = 0
                if (sys.version_info > (3, 0)):
                    raw_data = [' '.join([y.decode() for y in x]) for x in raw]
                else:
                    raw_data = [' '.join(x) for x in raw]
            else:
                raw_data, data = utils.perturb_sentence(
                    text, present, num_samples, self.neighbors, top_n=100,
                    use_proba=use_proba)
            labels = []
            if compute_labels:
                labels = (classifier_fn(raw_data) == true_label).astype(int)
            labels = np.array(labels)
            raw_data = np.array(raw_data).reshape(-1, 1)
            return raw_data, data, labels
        return words, positions, true_label, sample_fn

    def explain_instance(self, text, classifier_fn, threshold=0.95,
                          delta=0.1, tau=0.15, batch_size=100, use_proba=False,
                          beam_size=4,
                          **kwargs):
        words, positions, true_label, sample_fn = self.get_sample_fn(
            text, classifier_fn, use_proba=use_proba)
        # print words, true_label
        exp = anchor_base.AnchorBaseBeam.anchor_beam(
            sample_fn, delta=delta, epsilon=tau, batch_size=batch_size,
            desired_confidence=threshold, stop_on_first=True, **kwargs)
        exp['names'] = [words[x] for x in exp['feature']]
        exp['positions'] = [positions[x] for x in exp['feature']]
        exp['instance'] = text
        exp['prediction'] = true_label
        explanation = anchor_explanation.AnchorExplanation('text', exp,
                                                           self.as_html)
        return explanation
