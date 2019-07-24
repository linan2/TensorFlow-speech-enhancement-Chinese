#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017    Ke Wang

"""Converts data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import shutil
import sys

import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(sys.path[0]))
from kaldi_io import ArkReader
from tfrecords_io import make_sequence_example
from utils.misc import *

def make_config_file(name, output_dir, rt60_scp, inputs_scp, labels_scp=None):
    """Make temporal config file for making TFRecord files."""
    config_file = os.path.join(output_dir, "config_%s.list" % name)
    if labels_scp is not None:
        with open(rt60_scp, 'r') as fr_rt60, \
            open(inputs_scp, 'r') as fr_inputs, \
            open(labels_scp, 'r') as fr_labels, \
            open(config_file, 'w') as fw_config:
            for line_inputs in fr_inputs:
                line_rt60 = fr_rt60.readline()
                line_labels = fr_labels.readline()
                rt60_utt_id, rt60 = line_rt60.strip().split()
                inputs_utt_id, inputs_path = line_inputs.strip().split()
                labels_utt_id, labels_path = line_labels.strip().split()
                assert rt60_utt_id == inputs_utt_id
                assert inputs_utt_id == labels_utt_id
                line_config = inputs_utt_id + ' ' + inputs_path + ' ' \
                            + labels_path + ' ' + rt60 + '\n'
                fw_config.write(line_config)
    else:
        with open(rt60_scp, 'r') as fr_rt60, \
            open(inputs_scp, 'r') as fr_inputs, \
            open(config_file, 'w') as fw_config:
            for line_inputs in fr_inputs:
                line_rt60 = fr_rt60.readline()
                rt60_utt_id, rt60 = line_rt60.strip().split()
                inputs_utt_id, inputs_path = line_inputs.strip().split()
                assert rt60_utt_id == inputs_utt_id
                line_config = inputs_utt_id + ' ' + inputs_path + \
                    ' ' + rt60 + '\n'
                fw_config.write(line_config)
        # shutil.copyfile(inputs_scp, config_file)


def convert_to(name, rt60_scp, inputs_scp, labels_scp,
               output_dir, apply_cmvn=True, test=False):
    """Converts a dataset to tfrecords."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if apply_cmvn:
        cmvn = np.load(os.path.join(FLAGS.cmvn_dir, "train_cmvn.npz"))

    if test:
        make_config_file(name, output_dir, rt60_scp, inputs_scp)
    else:
        make_config_file(name, output_dir, rt60_scp, inputs_scp, labels_scp)

    config_file = os.path.join(output_dir, "config_%s.list" % name)
    tfrecords_name = os.path.join(output_dir, name + ".tfrecords")
    reader = ArkReader()
    with tf.python_io.TFRecordWriter(tfrecords_name) as writer, \
        open(config_file) as fr_config:
        for line in fr_config:
            if test:
                utt_id, inputs_path, rt60 = line.strip().split()
                inputs_path, inputs_offset = inputs_path.split(':')
            else:
                utt_id, inputs_path, labels_path, rt60 = line.strip().split()
                inputs_path, inputs_offset = inputs_path.split(':')
                labels_path, labels_offset = labels_path.split(':')

            tf.logging.info("Writing utterance %s to %s" % (
                utt_id, tfrecords_name))
            inputs = reader.read_ark(
                inputs_path, inputs_offset).astype(np.float64)
            # inputs = read_binary_file(
            #     inputs_path, inputs_offset).astype(np.float64)
            if test:
                labels = None
            else:
                labels = reader.read_ark(
                    labels_path, labels_offset).astype(np.float64)
                # labels = read_binary_file(
                #     labels_path, labels_offset).astype(np.float64)
            if apply_cmvn:
                inputs = (inputs - cmvn["mean_inputs"]) / cmvn["stddev_inputs"]
                frame_num = inputs.shape[0]
                rt60_numpy = np.ones(frame_num) * float(rt60)
                inputs = np.insert(inputs, 0, values=rt60_numpy, axis=1)
                if labels is not None:
                    labels = (labels - cmvn["mean_labels"]) / cmvn["stddev_labels"]
            ex = make_sequence_example(utt_id, inputs, labels)
            writer.write(ex.SerializeToString())

    os.remove(config_file)


def main(unused_argv):
    """Convert to Examples and write the result to TFRecords."""
    convert_to(FLAGS.name, FLAGS.rt60, FLAGS.inputs, FLAGS.labels,
              FLAGS.output_dir, FLAGS.apply_cmvn, FLAGS.test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--rt60',
        type=str,
        default='data/train/rt60.scp',
        help='File name of rt60 file.'
    )
    parser.add_argument(
        '--inputs',
        type=str,
        default='data/train/inputs.scp',
        help='File name of inputs.'
    )
    parser.add_argument(
        '--labels',
        type=str,
        default=None,
        help='File name of labels.'
    )
    parser.add_argument(
        '--cmvn_dir',
        type=str,
        default='data/train',
        help='Name of Numpy format CMVN file.'
    )
    parser.add_argument(
        '--apply_cmvn',
        type=str2bool,
        nargs='?',
        default='True',
        help='Whether apply CMVN to inputs and labels.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/tfrecords',
        help='Directory to write the converted result.'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='train',
        help="TFRecords name to save."
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help="Whether inputs is test file."
    )
    parser.add_argument(
        '--verbose',
        choices=[tf.logging.DEBUG,
                 tf.logging.ERROR,
                 tf.logging.FATAL,
                 tf.logging.INFO,
                 tf.logging.WARN],
        type=int,
        default=tf.logging.WARN,
        help="Log verbose."
    )
    FLAGS, unparsed = parser.parse_known_args()
    pp.pprint(FLAGS.__dict__)

    tf.logging.set_verbosity(FLAGS.verbose)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
