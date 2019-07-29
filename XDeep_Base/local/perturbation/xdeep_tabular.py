"""
Functions for explaining image classifiers.
"""
import util
import numpy as np
from shap import KernelExplainer
from anchor.anchor_tabular import AnchorTabularExplainer
from lime.lime_tabular import LimeTabularExplainer
from exceptions import XDeepError


class TabularExplainer(object):

    def __init__(self, predict_proba, train_data, train_labels, validation_data, validation_labels,
                 test_data, class_names, feature_names, categorical_features=None, categorical_names=None, discretizer='quartile'):
        self.explainers = {'lime': None, 'anchor': None, 'shap': None}
        self.explanations = {'lime': None, 'anchor': None, 'shap': None}
        self.predict_proba = predict_proba
        self.data = np.concatenate((train_data, validation_data, test_data), axis=0)
        self.train_data = train_data
        self.train_labels = train_labels
        self.validation_data = validation_data
        self.validation_labels = validation_labels
        self.test_data = test_data
        self.class_names = class_names
        self.feature_names = feature_names
        self.categorical_features = categorical_features
        self.categorical_names = categorical_names
        self.discretizer = discretizer
        self.__initialization()

    # Use default paramaters to initialize explainers
    def __initialization(self):
        print("Initialize default explainers")

        self.set_lime_paramaters(self.train_data, training_labels=self.train_labels
                                 , class_names=self.class_names, feature_names=self.feature_names
                                 , categorical_features=self.categorical_features
                                 , categorical_names=self.categorical_names)

        self.set_anchor_paramaters(self.class_names, self.feature_names, self.data, self.categorical_names
                                   , self.train_data, self.train_labels, self.validation_data
                                   , self.validation_labels)
        self.set_shap_paramaters()

    def set_lime_paramaters(self, training_data, discretizer=None, **kwargs):
        if discretizer is None:
            discretizer = self.discretizer
        self.explainers['lime'] = LimeTabularExplainer(training_data, discretizer=discretizer, **kwargs)

    def set_anchor_paramaters(self, class_names, feature_names, data, categorical_names, train_data,
                              train_labels, validation_data, validation_labels, discretizer=None, **kwargs):
        self.explainers['anchor'] = AnchorTabularExplainer(class_names, feature_names, data=data
                                                           , categorical_names=categorical_names)
        if discretizer is None:
            discretizer = self.discretizer
        self.explainers['anchor'].fit(train_data, train_labels, validation_data, validation_labels
                                      , discretizer=discretizer, **kwargs)

    def set_shap_paramaters(self):
        self.explainers['shap'] = KernelExplainer(self.predict_proba, self.train_data)

    def explain(self, instance, method='all', **kwargs):
        if method not in self.explainers and method != 'all':
            raise XDeepError("Please input correct explain_method:{} or 'all'".format(self.explainers.keys()))

        def predict_label(x):
            return np.argmax(self.predict_proba(x), axis=1)

        if method == 'lime':
            self.explanations['lime'] = self.explainers['lime']\
                .explain_instance(instance[:], self.predict_proba, **kwargs)
            return self.explanations['lime']
        if method == 'anchor':
            self.explanations['anchor'] = self.explainers['anchor']\
                .explain_instance(instance[:], predict_label, **kwargs)
            return self.explanations['anchor']
        if method == 'shap':
            self.explanations['shap'] = self.explainers['shap'].shap_values(instance[:], **kwargs)
            return self.explanations['shap']
        else:
            print("Explain with default paramaters")
            self.explanations['lime'] = self.explainers['lime'].explain_instance(instance[:], self.predict_proba)
            self.explanations['anchor'] = self.explainers['anchor'].explain_instance(instance[:], predict_label)
            self.explanations['shap'] = self.explainers['shap'].shap_values(instance[:], **kwargs)
            return self.explanations

    def get_explanation(self, method):
        if method not in self.explainers and method != 'all':
            raise XDeepError("Please input correct explain_method:{} or 'all'".format(self.explainers.keys()))
        if method == 'all':
            return self.explanations
        return self.explanations[method]

    def show_explanation(self, label):
        flag = False
        if self.explanations['lime'] is not None:
            util.show_lime_explanation(self.explanations['lime'], label)
            flag = True
        if self.explanations['anchor'] is not None:
            util.show_anchor_explanation(self.explanations['anchor'])
            flag = True
        if self.explanations['shap'] is not None:
            print(self.explanations['shap'])
            flag = True
        if not flag:
            print("You haven't get any explanation yet")


if __name__ == '__main__':
    pass