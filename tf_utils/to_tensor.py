#-*-coding:utf-8-*-
import numpy as np
import tensorflow as tf

def to_tensor(x):
    """Convert 2 dim Tensor to a 4 dim Tensor ready for convolution.

    Performs the opposite of flatten(x).  If the tensor is already 4-D, this
    returns the same as the input, leaving it unchanged.

    Parameters
    ----------
    x : tf.Tesnor
        Input 2-D tensor.  If 4-D already, left unchanged.

    Returns
    -------
    x : tf.Tensor
        4-D representation of the input.

    Raises
    ------
    ValueError
        If the tensor is not 2D or already 4D.
    """
    if len(x.get_shape()) == 2:
        n_input = x.get_shape().as_list()[1]
        x_dim = np.sqrt(n_input)
        if x_dim == int(x_dim):
            x_dim = int(x_dim)
            x_tensor = tf.reshape(
                x, [-1, x_dim, x_dim, 1], name='reshape')
        elif np.sqrt(n_input / 3) == int(np.sqrt(n_input / 3)):
            x_dim = int(np.sqrt(n_input / 3))
            x_tensor = tf.reshape(
                x, [-1, x_dim, x_dim, 3], name='reshape')
        else:
            x_tensor = tf.reshape(
                x, [-1, 1, 1, n_input], name='reshape')
    elif len(x.get_shape()) == 4:
        x_tensor = x
    else:
        raise ValueError('Unsupported input dimensions')
    return x_tensor