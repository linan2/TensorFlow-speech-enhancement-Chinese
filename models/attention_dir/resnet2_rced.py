#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017    Ke Wang


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer, l2_regularizer
from tensorflow.contrib.layers import batch_norm, fully_connected

class R_RCED(object):

    def __init__(self, rced):
        self.rced = rced

    def __call__(self, inputs, labels, reuse=False):
        """Build CNN models. On first pass will make vars."""
        self.inputs = inputs
        self.labels = labels
        print("-----------------------------inputs--------")
        print(np.shape(inputs))
        self.inputs_O = inputs
        outputs = self.infer(reuse)
        print(np.shape(outputs))
        return outputs

#    def CNN_Layer(inputs, filters_num, splice_dim, filters_width, activation_fn, normalizer_fn, normalizer_params, weights_regularizer):
#        tf.contrib.layers.conv2d(inputs, filters_num,
#                        [splice_dim, filters_width],
#                        activation_fn=activation_fn,
#                        normalizer_fn=normalizer_fn,
#                        normalizer_params=normalizer_params,
#                        weights_initializer=xavier_initializer(),
#                        weights_regularizer=weights_regularizer,
#                        biases_initializer=tf.zeros_initializer())
    def gru(inputs, num_units=None, bidirection=False, scope="gru", reuse=None):
        '''Applies a GRU.
        Args:
          inputs: A 3d tensor with shape of [N, T, C].
          num_units: An int. The number of hidden units.
          bidirection: A boolean. If True, bidirectional results 
            are concatenated.
          scope: Optional scope for `variable_scope`.  
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.
        Returns:
          If bidirection is True, a 3d tensor with shape of [N, T, 2*num_units],
            otherwise [N, T, num_units].
        '''
        with tf.variable_scope(scope, reuse=reuse):
            if num_units is None:
                num_units = inputs.get_shape().as_list[-1]
 
            cell = tf.contrib.rnn.GRUCell(num_units)  
            if bidirection: 
                cell_bw = tf.contrib.rnn.GRUCell(num_units)
                outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell, cell_bw, inputs, 
                                                         dtype=tf.float32)
                return tf.concat(outputs, 2)  
            else:
                outputs, _ = tf.nn.dynamic_rnn(cell, inputs, 
                                               dtype=tf.float32)
                return outputs

    def infer(self, reuse):
        rced = self.rced
        activation_fn = tf.nn.relu
        is_training = True

        input_dim = rced.input_dim
        left_context = rced.left_context
        right_context = rced.right_context
        splice_dim = left_context + 1 + right_context
        #inputs_O = self.inputs
        in_dims = self.inputs.get_shape().as_list()
        if len(in_dims) == 2:
            # shape format [batch, width]
            dims = self.inputs.get_shape().as_list()
            assert dims[0] == rced.batch_size
            inputs = tf.reshape(self.inputs, [dims[0], splice_dim, input_dim])
            inputs = tf.expand_dims(inputs, -1)
        elif len(in_dims) == 3:
            # shape format [batch, length, width]
            dims = self.inputs.get_shape().as_list()
            assert dims[0] == 1
            inputs = tf.squeeze(self.inputs, [0])
            inputs = tf.reshape(self.inputs, [-1, splice_dim, input_dim])
            inputs = tf.expand_dims(inputs, -1)

        # If test of cv , BN should use global mean / stddev
        if rced.cross_validation:
            is_training = False

        with tf.variable_scope('g_model') as scope:
            if reuse:
                scope.reuse_variables()

            if rced.batch_norm:
                normalizer_fn = batch_norm
                normalizer_params = {
                    "is_training": is_training,
                    "scale": True,
                    "renorm": True
                }
            else:
                normalizer_fn = None
                normalizer_params = None

            if rced.l2_scale > 0.0 and is_training:
                weights_regularizer = l2_regularizer(rced.l2_scale)
            else:
                weights_regularizer = None
                keep_prob = 1.0

            if not reuse:
                print("*** Generator summary ***")
                print("G inputs shape: {}".format(inputs.get_shape()))

            # inputs format [batch, in_height, in_width, in_channels]
            # filters format [filter_height, filter_width, in_channels, out_channels]
            filters_num = [12, 12, 24, 24, 32, 32, 24, 24, 12, 12]
            filters_width = [13, 11, 9, 7, 7, 7 ,7, 9, 11, 13]
            assert len(filters_num) == len(filters_num)
            #for i in range(len(filters_num)):
            #    inputs_0 = inputs
            #    inputs_name = "inputs_"+str(i)
            #    inpurts_name1 = "inputs_"+str(i+1)
            #    inputs_name1 = tf.contrib.layers.conv2d(inputs_name, filters_num[i],
            #            [splice_dim, filters_width[i]],
            #            activation_fn=activation_fn,
            #            normalizer_fn=normalizer_fn,
            #            normalizer_params=normalizer_params,
            #            weights_initializer=xavier_initializer(),
            #            weights_regularizer=weights_regularizer,
            #            biases_initializer=tf.zeros_initializer())
            #    inputs_name1 = inputs_name1 + inputs_name
                #if i > 0:
                #    inputs[i] = inputs[i] + inputs[i-1]

            #    if not reuse:
            #        print("Conv{} layer output shape: {}".format(
            #            i+1, inputs.get_shape()), end=" *** ")
            #        self.nnet_info(normalizer_fn, rced.keep_prob, weights_regularizer)
            #inputs_0 = CNN_Layer(inputs, filters_num[0], splice_dim, filters_width[0],activation_fn, 
            #                     normalizer_params, weights_regularizer,)
            inputs_O = tf.reshape(inputs, [-1,  splice_dim * input_dim])
            inputs_0 = tf.contrib.layers.conv2d(inputs, filters_num[0],[splice_dim, filters_width[0]],activation_fn=activation_fn,
                            normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,weights_initializer=xavier_initializer(),
                            weights_regularizer=weights_regularizer,biases_initializer=tf.zeros_initializer())
            #inputs_333 = inputs + inputs_0
            inputs_1 = tf.contrib.layers.conv2d(inputs_0, filters_num[1],[splice_dim, filters_width[1]],activation_fn=activation_fn,
                            normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,weights_initializer=xavier_initializer(),
                            weights_regularizer=weights_regularizer,biases_initializer=tf.zeros_initializer())
#            inputs_1 = inputs_1 + inputs_0
            inputs_2 = tf.contrib.layers.conv2d(inputs_1, filters_num[2],[splice_dim, filters_width[2]],activation_fn=activation_fn,
                            normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,weights_initializer=xavier_initializer(),
                            weights_regularizer=weights_regularizer,biases_initializer=tf.zeros_initializer())
            #inputs_2 = inputs_2 + inputs_1
            inputs_3 = tf.contrib.layers.conv2d(inputs_2, filters_num[3],[splice_dim, filters_width[3]],activation_fn=activation_fn,
                            normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,weights_initializer=xavier_initializer(),
                            weights_regularizer=weights_regularizer,biases_initializer=tf.zeros_initializer())
#            inputs_3 = inputs_3 + inputs_2
            inputs_4 = tf.contrib.layers.conv2d(inputs_3, filters_num[4],[splice_dim, filters_width[4]],activation_fn=activation_fn,
                            normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,weights_initializer=xavier_initializer(),
                            weights_regularizer=weights_regularizer,biases_initializer=tf.zeros_initializer())
            #inputs_4 = inputs_4 + inputs_3
            inputs_5 = tf.contrib.layers.conv2d(inputs_4, filters_num[5],[splice_dim, filters_width[5]],activation_fn=activation_fn,
                            normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,weights_initializer=xavier_initializer(),
                            weights_regularizer=weights_regularizer,biases_initializer=tf.zeros_initializer())
#            inputs_5 = inputs_5 + inputs_4
            inputs_6 = tf.contrib.layers.conv2d(inputs_5, filters_num[6],[splice_dim, filters_width[6]],activation_fn=activation_fn,
                            normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,weights_initializer=xavier_initializer(),
                            weights_regularizer=weights_regularizer,biases_initializer=tf.zeros_initializer())
            inputs_6 = inputs_6 + inputs_3
            inputs_7 = tf.contrib.layers.conv2d(inputs_6, filters_num[7],[splice_dim, filters_width[7]],activation_fn=activation_fn,
                            normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,weights_initializer=xavier_initializer(),
                            weights_regularizer=weights_regularizer,biases_initializer=tf.zeros_initializer())
#            inputs_7 = inputs_7 + inputs_6
            inputs_8 = tf.contrib.layers.conv2d(inputs_7, filters_num[8],[splice_dim, filters_width[8]],activation_fn=activation_fn,
                            normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,weights_initializer=xavier_initializer(),
                            weights_regularizer=weights_regularizer,biases_initializer=tf.zeros_initializer())
            inputs_8 = inputs_8 + inputs_1
            inputs_9 = tf.contrib.layers.conv2d(inputs_8, filters_num[9],[splice_dim, filters_width[9]],activation_fn=activation_fn,
                            normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,weights_initializer=xavier_initializer(),
                            weights_regularizer=weights_regularizer,biases_initializer=tf.zeros_initializer())
            #inputs_9 = inputs_9 + inputs_8
            print("***********shaper---------------------")
            print(np.shape(inputs_9))

#            name_I = "inputs_"+str(len(filters_num)+1)
#            inputs = name_I
            # Linear output
            # inputs = tf.reshape(inputs, [rced.batch_size, -1])
            inputs_D = tf.reshape(inputs_9, [-1, splice_dim * input_dim * filters_num[-1]])
            print("***********reshaper------------after---------")
            print(np.shape(inputs_D))

            yy_D = fully_connected(inputs_D, 13,
                                activation_fn=None,
                                weights_initializer=xavier_initializer(),
                                weights_regularizer=weights_regularizer,
                                biases_initializer=tf.zeros_initializer())
            print(np.shape(yy_D))

            inputs_D = tf.concat([inputs_D, inputs_O],1)
            y_0 = fully_connected(inputs_D, 1024,
                                activation_fn=activation_fn,
                                normalizer_fn=normalizer_fn,
                                weights_initializer=xavier_initializer(),
                                weights_regularizer=weights_regularizer,
                                biases_initializer=tf.zeros_initializer())
            print(np.shape(y_0))
            #y_0 = y_0 + inputs_D
            y_1 = fully_connected(y_0, 1024,
                                activation_fn=activation_fn,
                                normalizer_fn=normalizer_fn,
                                weights_initializer=xavier_initializer(),
                                weights_regularizer=weights_regularizer,
                                biases_initializer=tf.zeros_initializer())
            y = fully_connected(y_1, 15,
                                activation_fn=None,
                                weights_initializer=xavier_initializer(),
                                weights_regularizer=weights_regularizer,
                                biases_initializer=tf.zeros_initializer())
            y = tf.concat([y, yy_D],1)
            #y = fully_connected(inputs_D, rced.output_dim,
            #                    activation_fn=None,
            #                    weights_initializer=xavier_initializer(),
            #                    weights_regularizer=weights_regularizer,
            #                    biases_initializer=tf.constant_initializer(0.1))
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
            print("L2 regularizer scale is {}".format(self.rced.l2_scale),
                  end=" *** ")

        print()
