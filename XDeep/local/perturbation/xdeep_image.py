"""
Functions for explaining image classifiers.
"""
import util
from shap import KernelExplainer
from anchor.anchor_image import AnchorImage
from lime.lime_image import LimeImageExplainer
from exceptions import XDeepError


class ImageExplainer(object):

    def __init__(self, predict_proba, train_data=None):
        self.explainers = {'lime': None, 'anchor': None, 'shap': None}
        self.explanations = {'lime': None, 'anchor': None, 'shap': None}
        self.predict_proba = predict_proba
        self.train_data=train_data
        self.__initialization()

    # Use default paramaters to initialize explainers
    def __initialization(self):
        print("Initialize default explainers")
        self.set_lime_paramaters()
        self.set_anchor_paramaters()
        self.set_shap_paramaters()

    def set_lime_paramaters(self, **kwargs):
        self.explainers['lime'] = LimeImageExplainer(**kwargs)

    def set_anchor_paramaters(self, **kwargs):
        self.explainers['anchor'] = AnchorImage(**kwargs)

    def set_shap_paramaters(self):
        # self.explainers['shap'] = KernelExplainer(self.predict_proba, self.train_data)
        pass

    def explain(self, instance, method='all', **kwargs):
        if method not in self.explainers and method != 'all':
            raise XDeepError("Please input correct explain_method:{} or 'all'".format(self.explainers.keys()))

        if method == 'lime':
            self.explanations['lime'] = self.explainers['lime']\
                .explain_instance(instance[:], self.predict_proba, **kwargs)
            return self.explanations['lime']
        if method == 'anchor':
            self.explanations['anchor'] = self.explainers['anchor']\
                .explain_instance(instance[:], self.predict_proba, **kwargs)
            return self.explanations['anchor']
        if method == 'shap':
            self.explanations['shap'] = self.explainers['shap'].shap_values(instance[:], **kwargs)
            return self.explanations['shap']
        else:
            print("Explain with default paramaters")
            self.explanations['lime'] = self.explainers['lime'].explain_instance(instance[:], self.predict_proba)
            self.explanations['anchor'] = self.explainers['anchor'].explain_instance(instance[:], self.predict_proba)
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
            # util.show_lime_explanation(self.explanations['lime'])
            util.show_lime_image_explanation(self.explanations['lime'], label)
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