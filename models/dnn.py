#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017    Ke Wang

"""Build the feed forward fully connected neural networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, fully_connected
from tensorflow.contrib.layers import xavier_initializer, l2_regularizer


class DNN(object):

  def __init__(self, dnn):
    self.dnn = dnn

  def __call__(self, inputs, labels, reuse=False):
    """Build DNN model. On first pass will make vars."""
    self.inputs = inputs
    self.labels = labels
    outputs = self.infer(reuse)
    return outputs

  def infer(self, reuse):
    dnn = self.dnn
    units = 1024
    hidden_layers = 3
    activation_fn = tf.nn.relu

    in_dims = self.inputs.get_shape().as_list()
    if len(in_dims) == 2:
      # shape format [batch, width]
      dims = self.inputs.get_shape().as_list()
      inputs = self.inputs
    elif len(in_dims) == 3:
      # shape format [batch, length, width]
      dims = self.inputs.get_shape().as_list()
      assert dims[0] == 1
      inputs = tf.squeeze(self.inputs, axis=[0])

    # If test of cv , BN should use global mean / stddev
    is_training = False if dnn.cross_validation else True

    with tf.variable_scope('g_model') as scope:
      if reuse:
        scope.reuse_variables()

      if dnn.batch_norm:
        normalizer_fn = batch_norm
        normalizer_params = {
            "is_training": is_training,
            "scale": True,
            "renorm": True
        }
      else:
        normalizer_fn = None
        normalizer_params = None

      if dnn.l2_scale > 0.0 and is_training:
        weights_regularizer = l2_regularizer(dnn.l2_scale)
      else:
        weights_regularizer = None
        dnn.keep_prob = 1.0

      if not reuse:
        print("****************************************")
        print("*** Generator summary ***")
        print("G inputs shape: {}".format(inputs.get_shape()))
      sys.stdout.flush()

      h = fully_connected(inputs, units,
                          activation_fn=activation_fn,
                          normalizer_fn=normalizer_fn,
                          normalizer_params=normalizer_params,
                          weights_initializer=xavier_initializer(),
                          weights_regularizer=weights_regularizer,
                          biases_initializer=tf.zeros_initializer())
      h = self.dropout(h, dnn.keep_prob)
      if not reuse:
        print("G layer 1 output shape: {}".format(h.get_shape()), end=" *** ")
        self.nnet_info(normalizer_fn, dnn.keep_prob, weights_regularizer)

      for layer in range(hidden_layers):
        h = fully_connected(h, units,
                            activation_fn=activation_fn,
                            normalizer_fn=normalizer_fn,
                            normalizer_params=normalizer_params,
                            weights_initializer=xavier_initializer(),
                            weights_regularizer=weights_regularizer,
                            biases_initializer=tf.zeros_initializer())
        h = self.dropout(h, dnn.keep_prob)
        if not reuse:
          print("G layer {} output shape: {}".format(
              layer+2, h.get_shape()), end=" *** ")
          self.nnet_info(normalizer_fn, dnn.keep_prob, weights_regularizer)

      # Linear output
      y = fully_connected(h, dnn.output_dim,
                          activation_fn=None,
                          weights_initializer=xavier_initializer(),
                          weights_regularizer=weights_regularizer,
                          biases_initializer=tf.zeros_initializer())
      if not reuse:
        print("G output shape: {}".format(y.get_shape()))
        sys.stdout.flush()
    return y

  def dropout(self, x, keep_prob):
    if keep_prob != 1.0:
      y = tf.nn.dropout(x, keep_prob)
    else:
      y = x
    return y

  def nnet_info(self, batch_norm, keep_prob, weights_regularizer):
    if batch_norm is not None:
      print("use batch normalization", end=" *** ")
    if keep_prob != 1.0:
      print("keep prob is {}".format(keep_prob), end=" *** ")
    if weights_regularizer is not None:
      print("L2 regularizer scale is {}".format(self.dnn.l2_scale), end=" *** ")
    print()
