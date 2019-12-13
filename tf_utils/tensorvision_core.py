#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Core functions of TV."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import os
import numpy as np
import tensorflow as tf

from tf_utils import tensorvision_utils as utils


import pdb

flags = tf.app.flags
FLAGS = flags.FLAGS

tf.app.flags.DEFINE_boolean(
    'summary', True, ('Whether or not to save summaries to tensorboard.'))


def load_weights(checkpoint_dir, sess, saver):
    """
    Load the weights of a model stored in saver.

    Parameters
    ----------
    checkpoint_dir : str
        The directory of checkpoints.
    sess : tf.Session
        A Session to use to restore the parameters.
    saver : tf.train.Saver

    Returns
    -----------
    int
        training step of checkpoint
    """
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        logging.info(ckpt.model_checkpoint_path)
        file = os.path.basename(ckpt.model_checkpoint_path)
        checkpoint_path = os.path.join(checkpoint_dir, file)
        saver.restore(sess, checkpoint_path)
        return int(file.split('-')[1])


def build_training_graph(hypes, queue, modules):
    """
    Build the tensorflow graph out of the model files.

    Parameters
    ----------
    hypes : dict
        Hyperparameters
    queue: tf.queue
        Data Queue
    modules : tuple
        The modules load in utils.

    Returns
    -------
    tuple
        (q, train_op, loss, eval_lists) where
        q is a dict with keys 'train' and 'val' which includes queues,
        train_op is a tensorflow op,
        loss is a float,
        eval_lists is a dict with keys 'train' and 'val'
    """

    data_input = modules['input']
    encoder = modules['arch']
    objective = modules['objective']
    optimizer = modules['solver']

    learning_rate = tf.placeholder(tf.float32)

    # Add Input Producers to the Graph
    with tf.name_scope("Inputs"):
        image, labels = data_input.inputs(hypes, queue, phase='train')

    # Run inference on the encoder network
    logits = encoder.inference(hypes, image, train=True)

    # Build decoder on top of the logits
    decoded_logits = objective.decoder(hypes, logits, train=True)

    # Add to the Graph the Ops for loss calculation.
    with tf.name_scope("Loss"):
        losses = objective.loss(hypes, decoded_logits,
                                labels)

    # Add to the Graph the Ops that calculate and apply gradients.
    with tf.name_scope("Optimizer"):
        global_step = tf.Variable(0, trainable=False)
        # Build training operation
        train_op = optimizer.training(hypes, losses,
                                      global_step, learning_rate)

    with tf.name_scope("Evaluation"):
        # Add the Op to compare the logits to the labels during evaluation.
        eval_list = objective.evaluation(
            hypes, image, labels, decoded_logits, losses, global_step)

        summary_op = tf.summary.merge_all()

    graph = {}
    graph['losses'] = losses
    graph['eval_list'] = eval_list
    graph['summary_op'] = summary_op
    graph['train_op'] = train_op
    graph['global_step'] = global_step
    graph['learning_rate'] = learning_rate
    graph['decoded_logits'] = learning_rate

    return graph


def build_inference_graph(hypes, modules, image):
    """Run one evaluation against the full epoch of data.

    Parameters
    ----------
    hypes : dict
        Hyperparameters
    modules : tuble
        the modules load in utils
    image : placeholder

    return:
        graph_ops
    """
    with tf.name_scope("Validation"):

        logits = modules['arch'].inference(hypes, image, train=False)

        decoded_logits = modules['objective'].decoder(hypes, logits,
                                                      train=False)
    return decoded_logits, logits


def start_tv_session(hypes):
    """
    Run one evaluation against the full epoch of data.

    Parameters
    ----------
    hypes : dict
        Hyperparameters

    Returns
    -------
    tuple
        (sess, saver, summary_op, summary_writer, threads)
    """
    # Build the summary operation based on the TF collection of Summaries.
    if FLAGS.summary:
        tf.contrib.layers.summarize_collection(tf.GraphKeys.WEIGHTS)
        tf.contrib.layers.summarize_collection(tf.GraphKeys.BIASES)
        summary_op = tf.summary.merge_all()
    else:
        summary_op = None

    # Create a saver for writing training checkpoints.
    if 'keep_checkpoint_every_n_hours' in hypes['solver']:
        kc = hypes['solver']['keep_checkpoint_every_n_hours']
    else:
        kc = 10000.0

    saver = tf.train.Saver(max_to_keep=int(utils.cfg.max_to_keep))

    sess = tf.get_default_session()

# for VGG / darknet53
#    ckpt = tf.train.get_checkpoint_state(hypes['dirs']['data_dir'])
#    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
#        saver.restore(sess, ckpt.model_checkpoint_path) 
#        print("continue training ...")
#    else:
#        sess.run(tf.global_variables_initializer()) 
#        print("initiate training ...")

# for mobilenet
#    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='MobileNet')
#    var_to_restore = [val for val in var_list if 'score' not in val.name and 'fc_16' not in val.name]
#    saver_restore = tf.train.Saver(var_to_restore)
#    print(['var list length: ',len(var_to_restore)])
#    # Run the Op to initialize the variables.    
#    sess.run(tf.global_variables_initializer()) 
#    ckpt = tf.train.get_checkpoint_state('./pretrained/mobilenet/')
#    if ckpt:
#        saver_restore.restore(sess, ckpt.model_checkpoint_path) 
#        print("Pretrained weights restored ...")

# for deeplab v3
#    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='resnet_v2_50')
#    var_to_restore = [val for val in var_list if 'logits' not in val.name]
#    saver_restore = tf.train.Saver(var_to_restore)
#    sess.run(tf.global_variables_initializer())
#    saver_restore.restore(sess, './pretrained/deeplab_v3/resnet_v2_50.ckpt')
#    print('Resnet50 pretrained weights restored ...')

# for mobilenet v2
#    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='MobilenetV2')
#    var_to_restore = [val for val in var_list if 'Logits' not in val.name]
#    #var_to_restore = [val for val in var_list if 'Logits' not in val.name and 'MobilenetV2/Conv/' not in val.name] # remove weights of first layer
#    saver_restore = tf.train.Saver(var_to_restore)
#    sess.run(tf.global_variables_initializer())
#    saver_restore.restore(sess, './pretrained/mobilenet_v2/mobilenet_v2_1.0_224.ckpt')
#    print('Mobilenetv2 Pretrained weights restored ...')

# for mobilenet v2 continue training
#    saver.restore(sess, './RUNS/test/seg_bdd100k_MobileNet_V2_seg_2018_09_11_17.15/model.ckpt-49999')
#    print('Continue training from BDD segmentation dataset ...')

# for BiSeNet_resnet
#    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='resnet_v2_101')
#    var_to_restore = [val for val in var_list if 'logits' not in val.name]
#    saver_restore = tf.train.Saver(var_to_restore)
#    sess.run(tf.global_variables_initializer())
#    saver_restore.restore(sess, './pretrained/resnet_v2_101.ckpt')
#    print('BiSeNet frontend "ResNet101" weights restored ...')

# for BiSeNet continue training
    saver.restore(sess, './RUNS/test/__bdd100k_BiSeNet_drivable_diff_2018_11_10_19.18/model.ckpt-699999')
    print('BiSeNet Pretrained weights restored ...')

# for BiSeNet_MobileNetV2
#    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='MobilenetV2')
#    var_to_restore = [val for val in var_list if 'logits' not in val.name]
#    saver_restore = tf.train.Saver(var_to_restore)
#    sess.run(tf.global_variables_initializer())
#    saver_restore.restore(sess, './pretrained/mobilenet_v2/mobilenet_v2_1.0_224.ckpt')
#    print('BiSeNet frontend "Mobilenetv2" weights restored ...')

    # Start the queue runners.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.summary.FileWriter(hypes['dirs']['output_dir'],
                                           graph=sess.graph)

    tv_session = {}
    tv_session['sess'] = sess
    tv_session['saver'] = saver
    tv_session['summary_op'] = summary_op
    tv_session['writer'] = summary_writer
    tv_session['coord'] = coord
    tv_session['threads'] = threads

    return tv_session
