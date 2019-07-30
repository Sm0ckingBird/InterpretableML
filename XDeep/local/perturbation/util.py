import shap
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from nets import inception
from preprocessing import inception_preprocessing
from datasets import imagenet
slim = tf.contrib.slim
sys.path.append('/Users/zhangzijian/models/research/slim')

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

def image_tutorial():
    session = tf.Session()
    image_size = inception.inception_v3.default_image_size
    def transform_img_fn(path_list):
        out = []
        for f in path_list:
            image_raw = tf.image.decode_jpeg(open(f,'rb').read(), channels=3)
            image = inception_preprocessing.preprocess_image(image_raw, image_size, image_size, is_training=False)
            out.append(image)
        return session.run([out])[0]
    names = imagenet.create_readable_names_for_imagenet_labels()
    processed_images = tf.placeholder(tf.float32, shape=(None, 299, 299, 3))
    with slim.arg_scope(inception.inception_v3_arg_scope()):
        logits, _ = inception.inception_v3(processed_images, num_classes=1001, is_training=False)
    probabilities = tf.nn.softmax(logits)
    init_fn = slim.assign_from_checkpoint_fn(
        '/Users/zhangzijian/22j/Projects/DeepLearningInterpreter/Material/Code/lime-master/tf-models-master/slim/pretrained/inception_v3.ckpt',
        slim.get_model_variables('InceptionV3'))
    init_fn(session)
    def predict_fn(images):
        return session.run(probabilities, feed_dict={processed_images: images})
    
    images = transform_img_fn(['dogs.jpg'])
    # I'm dividing by 2 and adding 0.5 because of how this Inception represents images
    plt.imshow(images[0] / 2 + 0.5)
    preds = predict_fn(images)
    for x in preds.argsort()[0][-5:]:
        print(x, names[x], preds[0,x])
    image = images[0]
    return predict_fn, image

