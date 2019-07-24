#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017    Ke Wang

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer, l2_regularizer
from tensorflow.contrib.layers import batch_norm, fully_connected

class CNN(object):

    def __init__(self, cnn):
        self.cnn = cnn

    def __call__(self, inputs, labels, reuse=False):
        """Build CNN models. On first pass will make vars."""
        self.inputs = inputs
        self.labels = labels

        outputs = self.infer(reuse)

        return outputs

    def infer(self, reuse):
        cnn = self.cnn
        activation_fn = tf.nn.relu
        is_training = True

        input_dim = cnn.input_dim
        left_context = cnn.left_context
        right_context = cnn.right_context
        splice_dim = left_context + 1 + right_context

        in_dims = self.inputs.get_shape().as_list()
        if len(in_dims) == 2:
            # shape format [batch, width]
            dims = self.inputs.get_shape().as_list()
            assert dims[0] == cnn.batch_size
            inputs = tf.reshape(self.inputs, [dims[0], splice_dim, input_dim])
            inputs = tf.expand_dims(inputs, -1)
        elif len(in_dims) == 3:
            # shape format [batch, length, width]
            dims = self.inputs.get_shape().as_list()
            assert dims[0] == 1
            inputs = tf.squeeze(self.inputs, [0])
            inputs = tf.reshape(inputs, [-1, splice_dim, input_dim])
            inputs = tf.expand_dims(inputs, -1)

        # If test of cv , BN should use global mean / stddev
        if cnn.cross_validation:
            is_training = False

        with tf.variable_scope('g_model') as scope:
            if reuse:
                scope.reuse_variables()

            if cnn.batch_norm:
                normalizer_fn = batch_norm
                normalizer_params = {
                    "is_training": is_training,
                    "scale": True,
                    "renorm": True
                }
            else:
                normalizer_fn = None
                normalizer_params = None

            if cnn.l2_scale > 0.0 and is_training:
                weights_regularizer = l2_regularizer(cnn.l2_scale)
            else:
                weights_regularizer = None
                keep_prob = 1.0

            if not reuse:
                print("*** Generator summary ***")
                print("G inputs shape: {}".format(inputs.get_shape()))

            # conv1
            # inputs format [batch, in_height, in_width, in_channels]
            # filters format [filter_height, filter_width, in_channels, out_channels]
            filter_num = [32, 64]
            filter_width = [11, 11]
            assert len(filters_num) == len(filters_num)
            for i in range(len(filters_num)):
                inputs = tf.contrib.layers.conv2d(inputs, filters_num[i],
                        [splice_dim, filters_width[i]],
                        activation_fn=activation_fn,
                        normalizer_fn=normalizer_fn,
                        normalizer_params=normalizer_params,
                        weights_initializer=xavier_initializer(),
                        weights_regularizer=weights_regularizer,
                        biases_initializer=tf.zeros_initializer())
                if not reuse:
                    print("Conv{} layer output shape: {}".format(
                        i+1, inputs.get_shape()), end=" *** ")
                    self.nnet_info(normalizer_fn, rced.keep_prob, weights_regularizer)

            # kernel = tf.get_variable('weights_1', [11, 11, 1, 32],
            #     initializer=tf.truncated_normal_initializer(stddev=0.05),
            #     regularizer=weights_regularizer)
            # conv = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
            # biases = tf.get_variable('biases_1', [32],
            #     initializer=tf.constant_initializer(0.1))
            # pre_activation = tf.nn.bias_add(conv, biases)
            # if cnn.batch_norm:
            #     pre_activation = batch_norm(pre_activation, scale=True,
            #                                 is_training=is_training,
            #                                 renorm=True)
            # h = tf.nn.relu(pre_activation)
            # # pool1
            # # pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1],
            # #                        strides=[1, 2, 2, 1],
            # #                        padding='SAME')
            # if not reuse:
            #     print("Conv1 layer output shape: {}".format(h.get_shape()),
            #           end=" *** ")
            #     self.nnet_info(normalizer_fn, cnn.keep_prob, weights_regularizer)

            # # conv2
            # kernel = tf.get_variable('weights_2', [11, 11, 32, 64],
            #     initializer=tf.truncated_normal_initializer(stddev=0.05),
            #     regularizer=weights_regularizer)
            # conv = tf.nn.conv2d(h, kernel, [1, 1, 1, 1], padding='SAME')
            # biases = tf.get_variable('biases_2', [64],
            #     initializer=tf.constant_initializer(0.1))
            # pre_activation = tf.nn.bias_add(conv, biases)
            # if cnn.batch_norm:
            #     pre_activation = batch_norm(pre_activation, scale=True,
            #                                 is_training=is_training,
            #                                 renorm=True)
            # h = tf.nn.relu(pre_activation)
            # # pool2
            # # pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1],
            # #                        strides=[1, 2, 2, 1],
            # #                        padding='SAME')
            # if not reuse:
            #     print("Conv2 layer output shape: {}".format(h.get_shape()),
            #           end=" *** ")
            #     self.nnet_info(normalizer_fn, cnn.keep_prob, weights_regularizer)

            # local3
            # Move everything into depth so we can perform a single matrix multiply.
            # reshape = tf.reshape(h, [cnn.batch_size, -1])
            reshape = tf.reshape(inputs, [-1, splice_dim * input_dim * filters_num[-1]])
            h = fully_connected(reshape, 512,
                                activation_fn=activation_fn,
                                normalizer_fn=normalizer_fn,
                                normalizer_params=normalizer_params,
                                weights_initializer=xavier_initializer(),
                                weights_regularizer=weights_regularizer,
                                biases_initializer=tf.zeros_initializer())
            if not reuse:
                print("Local3 layer output shape: {}".format(h.get_shape()),
                      end=" *** ")
                self.nnet_info(normalizer_fn, cnn.keep_prob, weights_regularizer)

            # local4
            h = fully_connected(h, 512,
                                activation_fn=activation_fn,
                                normalizer_fn=normalizer_fn,
                                normalizer_params=normalizer_params,
                                weights_initializer=xavier_initializer(),
                                weights_regularizer=weights_regularizer,
                                biases_initializer=tf.constant_initializer(0.1))
            if not reuse:
                print("Local4 layer output shape: {}".format(h.get_shape()),
                      end=" *** ")
                self.nnet_info(normalizer_fn, cnn.keep_prob, weights_regularizer)

            # Linear output
            y = fully_connected(h, cnn.output_dim,
                                activation_fn=None,
                                weights_initializer=xavier_initializer(),
                                weights_regularizer=weights_regularizer,
                                biases_initializer=tf.constant_initializer(0.1))
            if not reuse:
                print("G output shape: {}".format(y.get_shape()))
                sys.stdout.flush()

        return y

    def nnet_info(self, batch_norm, keep_prob, weights_regularizer):
        if batch_norm is not None:
            print("use batch normalization", end=" *** ")
        if keep_prob != 1.0:
            print("keep prob is {}".format(keep_prob),
                  end=" *** ")
        if weights_regularizer is not None:
            print("L2 regularizer scale is {}".format(self.cnn.l2_scale),
                  end=" *** ")

        print()
