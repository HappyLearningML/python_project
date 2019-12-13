#-*-coding:utf-8-*-
import tensorflow as tf
import numpy as np
import scipy.io

def load_net(data_path):
    data = scipy.io.loadmat(data_path)
    if 'normalization' in data:
        # old format, for data where
        # MD5(imagenet-vgg-verydeep-19.mat) = 8ee3263992981a1d26e73b3ca028a123
        mean_pixel = np.mean(data['normalization'][0][0][0], axis=(0, 1))
    else:
        # new format, for data where
        # MD5(imagenet-vgg-verydeep-19.mat) = 106118b7cf60435e6d8e04f6a6dc3657
        mean_pixel = data['meta']['normalization'][0][0][0][0][2][0][0]
    weights = data['layers'][0]
    return weights, mean_pixel


def load_graph(frozen_graph_filename):
    """Load a (frozen) Tensorflow model into memory."""
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(frozen_graph_filename, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return graph