#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019.1    Nan LEE

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np

sys.path.append(os.path.dirname(sys.path[0]))
from models.dnn import *
from models.cnn import *
from models.rced import *
from models.u_net import *
from utils.ops import *
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
        # config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

class Model(object):

    def __init__(self, name='BaseModel'):
        self.name = name

    def save(self, save_dir, step):
        model_name = self.name
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not hasattr(self, 'saver'):
            self.saver = tf.train.Saver()
        self.saver.save(self.sess,
                        os.path.join(save_dir, model_name),
                        global_step=step)

    def load(self, save_dir, model_file=None, moving_average=False):
        if not os.path.exists(save_dir):
            print('[!] Checkpoints path does not exist...')
            return False
        print('[*] Reading checkpoints...')
        if model_file is None:
            ckpt = tf.train.get_checkpoint_state(save_dir)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            else:
                return False
        else:
            ckpt_name = model_file

        if moving_average:
            # Restore the moving average version of the learned variables for eval.
            variable_averages = tf.train.ExponentialMovingAverage(
                                                     self.MOVING_AVERAGE_DECAY)
            variables_to_restore = variable_averages.variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)
        else:
            saver = tf.train.Saver()
        saver.restore(self.sess, os.path.join(save_dir, ckpt_name))
        print('[*] Read {}'.format(ckpt_name))
        return True


class DNNTrainer(Model):
    """Generative Adversarial Network for Speech Enhancement"""
    def __init__(self, sess, args, devices,
                 inputs, labels, cross_validation=False, name='DNNTrainer'):
        super(DNNTrainer, self).__init__(name)
        self.sess = sess
        self.cross_validation = cross_validation
        self.MOVING_AVERAGE_DECAY = 0.9999
        if cross_validation:
            self.keep_prob = 1.0
        else:
            self.keep_prob = args.keep_prob
        self.batch_norm = args.batch_norm
        self.batch_size = args.batch_size
        self.devices = devices
        self.save_dir = args.save_dir
        self.writer = tf.summary.FileWriter(os.path.join(
            args.save_dir,'train'), sess.graph)
        self.l2_scale = args.l2_scale
        # data
        self.input_dim = args.input_dim
        self.output_dim = args.output_dim
        self.left_context = args.left_context
        self.right_context = args.right_context
        self.batch_size = args.batch_size
        # Batch Normalization
        self.batch_norm = args.batch_norm
        self.g_disturb_weights = False
        # define the functions
        self.g_learning_rate = tf.Variable(args.g_learning_rate, trainable=False)
        if args.g_type == 'dnn':
            self.generator = DNN(self)
        elif args.g_type == 'cnn':
            self.generator = CNN(self)
        elif args.g_type == 'res_rced':
            self.generator = R_RCED(self)
        else:
            raise ValueError('Unrecognized G type {}'.format(args.g_type))
        if labels is None:
            self.generator(inputs, labels, reuse=False)
        else:
            self.build_model(inputs, labels)

    def build_model(self, inputs, labels):
        all_g_grads = []
        # g_opt = tf.train.RMSPropOptimizer(self.g_learning_rate)
        # g_opt = tf.train.GradientDescentOptimizer(self.g_learning_rate)
        g_opt = tf.train.AdamOptimizer(self.g_learning_rate)
        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
                self.MOVING_AVERAGE_DECAY)

        with tf.variable_scope(tf.get_variable_scope()):
            for idx, device in enumerate(self.devices):
                with tf.device("/%s" % device):
                    with tf.name_scope("device_%s" % idx):
                        with variables_on_gpu():
                            self.build_model_single_gpu(idx, inputs, labels)
                            tf.get_variable_scope().reuse_variables()
                            if not self.cross_validation:
                                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                                with tf.control_dependencies(update_ops):
                                    g_grads = g_opt.compute_gradients(
                                        self.g_losses[-1], var_list=self.g_vars)
                                    all_g_grads.append(g_grads)
        if not self.cross_validation:
            avg_g_grads = average_gradients(all_g_grads)
            g_apply_gradient_op = g_opt.apply_gradients(avg_g_grads)
            variables_averages_op = variable_averages.apply(
                    tf.trainable_variables())
            # Group all updates to into a single train op.
            self.g_opt = tf.group(g_apply_gradient_op, variables_averages_op)


    def build_model_single_gpu(self, gpu_idx, inputs, labels):
        if gpu_idx == 0:
            g = self.generator(inputs, labels, reuse=False)

        g = self.generator(inputs, labels, reuse=True)

        if gpu_idx == 0:
            self.g_losses = []
            self.g_mse_losses = []
            self.g_l2_losses = []
        #g_mse_lossm = 0.5 * tf.losses.mean_squared_error(g_m, label_m) * 12

        g_mse_loss = tf.losses.mean_squared_error(g, labels) 
        #g_mse_loss2 = 0.5 * tf.losses.mean_squared_error(g_p, label_p) * 15
        if not self.cross_validation and self.l2_scale > 0.0:
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, '.*g_model')
            g_l2_loss = tf.reduce_sum(reg_losses)
        else:
            g_l2_loss = tf.constant(0.0)
        g_loss = g_mse_loss + g_l2_loss

        self.g_mse_losses.append(g_mse_loss)
        self.g_l2_losses.append(g_l2_loss)
        self.g_losses.append(g_loss)

        self.g_mse_loss_summ = scalar_summary("g_mse_loss", g_mse_loss)
        self.g_l2_loss_summ = scalar_summary("g_l2_loss", g_l2_loss)
        self.g_loss_summ = scalar_summary("g_loss", g_loss)

        summaries = [self.g_mse_loss_summ,
                     self.g_l2_loss_summ,
                     self.g_loss_summ]

        self.summaries = tf.summary.merge(summaries)

        if gpu_idx == 0:
            self.get_vars()

    def get_vars(self):
        t_vars = tf.trainable_variables()
        self.g_vars_dict = {}
        for var in t_vars:
            if var.name.startswith('g_'):
                self.g_vars_dict[var.name] = var
        self.g_vars = self.g_vars_dict.values()
        self.all_vars = t_vars
        if self.g_disturb_weights and not self.cross_validation:
            stddev = 0.00001
            print("Add Gaussian noise to G weights (stddev = %s)" % (stddev))
            sys.stdout.flush()
            self.g_disturb = [v.assign(
                tf.add(v, tf.truncated_normal([], 0, stddev))) for v in self.g_vars]
        else:
            print("Not add noise to G weights")
