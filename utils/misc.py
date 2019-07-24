#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017    Ke Wang     Xiaomi

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import pprint

import tensorflow as tf
import tensorflow.contrib.slim as slim


pp = pprint.PrettyPrinter()

def check_tensorflow_version():
    if tf.__version__ < "1.3.0":
        raise EnvironmentError("Tensorflow version must >= 1.3.0")
    else:
        print(tf.__version__)


def read_list(filename):
    data_list = []
    with open(filename, 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            line = line.strip()
            data_list.append(line)
    return data_list


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)
    sys.stdout.flush()


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
