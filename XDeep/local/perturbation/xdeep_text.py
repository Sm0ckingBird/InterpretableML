"""
Functions for explaining text classifiers.
"""
import spacy
import util
import numpy as np
from exceptions import XDeepError
from anchor.anchor_text import AnchorText
from lime.lime_text import LimeTextExplainer


class TextExplainer(object):

    def __init__(self, class_names):
        self.explainers = {'lime': None, 'anchor': None}
        self.explanations = {'lime': None, 'anchor': None}
        self.class_names = class_names
        self.__initialization(class_names)

    # Use default paramaters to initialize explainers
    def __initialization(self, class_names=None):
        print("Initialize default explainers")
        self.set_lime_paramaters(class_names=class_names)
        self.set_anchor_paramaters(None, class_names)

    def set_lime_paramaters(self, class_names=None, **kwargs):
        if class_names is None:
            class_names = self.class_names
        self.explainers['lime'] = LimeTextExplainer(class_names=class_names, **kwargs)

    def set_anchor_paramaters(self, nlp, class_names=None, **kwargs):
        if class_names is None:
            class_names = self.class_names
        if nlp is None:
            print("Use default nlp = spacy.load('en_core_web_lg')")
            nlp = spacy.load('en_core_web_lg')
        self.explainers['anchor'] = AnchorText(nlp, class_names, **kwargs)

    def explain(self, instance, predict_proba, method='all', **kwargs):
        if method not in self.explainers and method != 'all':
            raise XDeepError("Please input correct explain_method:{} or 'all'".format(self.explainers.keys()))

        def predict_label(x):
            return np.argmax(predict_proba(x), axis=1)

        if method == 'lime':
            self.explanations['lime'] = self.explainers['lime']\
                .explain_instance(str(instance), predict_proba, **kwargs)
            return self.explanations['lime']
        if method == 'anchor':
            self.explanations['anchor'] = self.explainers['anchor']\
                .explain_instance(str(instance), predict_label, **kwargs)
            return self.explanations['anchor']
        else:
            print("Explain with default paramaters")
            self.explanations['lime'] = self.explainers['lime'].explain_instance(str(instance), predict_proba)
            self.explanations['anchor'] = self.explainers['anchor'].explain_instance(str(instance), predict_label)
            return self.explanations

    def get_explanation(self, method):
        if method not in self.explainers and method != 'all':
            raise XDeepError("Please input correct explain_method:{} or 'all'".format(self.explainers.keys()))
        if method == 'all':
            return self.explanations
        return self.explanations[method]

    def show_explanation(self):
        flag = False
        if self.explanations['lime'] is not None:
            util.show_lime_explanation(self.explanations['lime'])
            flag = True
        if self.explanations['anchor'] is not None:
            util.show_anchor_explanation(self.explanations['anchor'])
            flag = True
        if not flag:
            print("You haven't get any explanation yet")


if __name__ == '__main__':
    pass
