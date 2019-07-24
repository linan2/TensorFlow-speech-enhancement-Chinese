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
from tfrecords_io import make_sequence_example
from utils.misc import *
import cPickle
def convert_to(name, inputs_scp,
               output_dir, test=False):
    """Converts a dataset to tfrecords."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tfrecords_name = os.path.join(output_dir, name + ".tfrecords")
    con = 0
    with tf.python_io.TFRecordWriter(tfrecords_name) as writer, \
        open(inputs_scp,'r') as inputs_f:
        for line in inputs_f:
            #print(line)
            line = line.split()
            utt_id = line[0] 
            #print(utt_id)
            inputs_path = line[-1]
            data = cPickle.load(open(inputs_path, 'rb'))
            [mixed_complx_x] = data
            inputs = mixed_complx_x
            inputs = np.abs(inputs)
            inputs = np.log(inputs + 1e-08).astype(np.float32)
            inputs = inputs - np.mean(inputs,0)
            #print(np.shape(inputs))
            #print(speech_x)
            #labels = speech_x
            #print(np.shape(labels))
            tf.logging.info("Writing utterance %s to %s" % (
                utt_id, tfrecords_name))
            ex = make_sequence_example(utt_id, inputs)
            writer.write(ex.SerializeToString())
            con += 1
            print("done write")
            print(con)
    #os.remove(config_file)
    print("done")

def main(unused_argv):
    """Convert to Examples and write the result to TFRecords."""
    convert_to(FLAGS.name, FLAGS.inputs,
              FLAGS.output_dir, FLAGS.test)
    print("done all")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--inputs',
        type=str,
        default='data/train/inputs.scp',
        help='File name of inputs.'
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
