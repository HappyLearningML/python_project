#-*-coding:utf-8-*-
import tensorflow as tf

def weight_variable(shape, **kwargs):
    '''Helper function to create a weight variable initialized with
    a normal distribution
    Parameters
    ----------
    shape : list
        Size of weight variable
    '''
    if isinstance(shape, list):
        initial = tf.random_normal(tf.stack(shape), mean=0.0, stddev=0.01)
        initial.set_shape(shape)
    else:
        initial = tf.random_normal(shape, mean=0.0, stddev=0.01)
    return tf.Variable(initial, **kwargs)



def bias_variable(shape, **kwargs):
    '''Helper function to create a bias variable initialized with
    a constant value.
    Parameters
    ----------
    shape : list
        Size of weight variable
    '''
    if isinstance(shape, list):
        initial = tf.random_normal(tf.stack(shape), mean=0.0, stddev=0.01)
        initial.set_shape(shape)
    else:
        initial = tf.random_normal(shape, mean=0.0, stddev=0.01)
    return tf.Variable(initial, **kwargs)