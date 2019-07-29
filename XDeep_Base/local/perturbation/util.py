import shap
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt


def show_lime_explanation(exp, show_in_note_book=True):
	print("-------------------Lime Explanation-------------------")
	print(exp.as_list())
	exp.as_pyplot_figure()
	if show_in_note_book:
		exp.show_in_notebook(text=True)


def show_lime_image_explanation(exp, label, positive_only=True, num_features=5, hide_rest=False):
	temp, mask = exp.get_image_and_mask(label, positive_only=positive_only,
										num_features=num_features, hide_rest=hide_rest)
	plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))


def show_anchor_explanation(exp, show_in_note_book=True, verbose=False):
	if verbose:
		print()
		print('Examples where anchor applies and model predicts same to instance:')
		print()
		print('\n'.join([x[0] for x in exp.examples(only_same_prediction=True)]))
		print()
		print('Examples where anchor applies and model predicts different with instance:')
		print()
		print('\n'.join([x[0] for x in exp.examples(only_different_prediction=True)]))
	print("-------------------Anchor Explanation-------------------")
	print('Anchor: %s' % (' AND '.join(exp.names())))
	print('Precision: %.2f' % exp.precision())
	if show_in_note_book:
		exp.show_in_notebook()


def show_shap_explanation(expected_value, shap_value, X):
	shap.force_plot(expected_value, shap_value, X)