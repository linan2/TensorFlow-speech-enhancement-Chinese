#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017    Ke Wang     Xiaomi

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
import os.path
import sys
import time

import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(sys.path[0]))
from tfrecords_io import get_padded_batch, get_batch
from utils.misc import pp

tf.logging.set_verbosity(tf.logging.INFO)


def main():
    # names = ['train1']
    names = ['tr1', 'tr2', 'tr3', 'tr4', 'tr5']
    # names = ['train']
    tfrecords_lst = []
    for name in names:
        tfrecords_name = os.path.join(FLAGS.data_dir, name + ".tfrecords")
        tfrecords_lst.append(tfrecords_name)
    tf.logging.info(tfrecords_lst)

    with tf.Graph().as_default():
        # utt_id, inputs, labels, lengths = get_padded_batch(
        #     tfrecords_lst, FLAGS.batch_size, FLAGS.input_dim,
        #     FLAGS.output_dim, 0, 0,
        #     num_enqueuing_threads=FLAGS.num_threads,
        #     num_epochs=FLAGS.num_epochs,
        #     infer=False)
        # utt_id, inputs, lengths = get_padded_batch(
        #     tfrecords_lst, FLAGS.batch_size, FLAGS.input_dim,
        #     FLAGS.output_dim, 1, 1,
        #     num_enqueuing_threads=FLAGS.num_threads,
        #     num_epochs=FLAGS.num_epochs,
        #     infer=True)
        inputs, labels = get_batch(
            tfrecords_lst, FLAGS.batch_size, FLAGS.input_dim,
            FLAGS.output_dim, 5, 5, num_enqueuing_threads=FLAGS.num_threads,
            num_epochs=FLAGS.num_epochs)
        # print(inputs.get_shape().as_list())
        # utt_id, inputs, lengths = get_batch(
        #     tfrecords_lst, FLAGS.batch_size, FLAGS.input_dim,
        #     FLAGS.output_dim, 3, 3, num_enqueuing_threads=FLAGS.num_threads,
        #     num_epochs=FLAGS.num_epochs, infer=True)


        init = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())

        sess = tf.Session()

        sess.run(init)

        start = datetime.datetime.now()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            batch = 0
            while not coord.should_stop():
                # Print an overview fairly often.
                # tr_utt_id, tr_inputs, tr_labels, tr_lengths = sess.run([
                #     utt_id, inputs, labels, lengths])
                tr_inputs, tr_labels = sess.run([
                    inputs, labels])
                # tr_utt_id, tr_inputs, tr_lengths = sess.run([
                #     utt_id, inputs, lengths])
                # tf.logging.info(tr_utt_id)
                # tf.logging.info(tr_inputs)
                # tf.logging.info(tr_labels)
                # tf.logging.info('inputs shape : '+ str(tr_inputs.shape))
                # tf.logging.info('labels shape : ' + str(tr_labels.shape))
                # tf.logging.info('actual lengths : ' + str(tr_lengths))
                batch += 1
        except tf.errors.OutOfRangeError:
            tf.logging.info("Batch number is %d" % batch)
            tf.logging.info('Done training -- epoch limit reached')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        end = datetime.datetime.now()
        duration = (end - start).total_seconds()
        print("Reading time is %.0fs." % duration)
        # Wait for threads to finish.
        coord.join(threads)
        sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Mini-batch size.'
    )
    parser.add_argument(
        '--input_dim',
        type=int,
        default=257,
        help='The dimension of inputs.'
    )
    parser.add_argument(
        '--output_dim',
        type=int,
        default=40,
        help='The dimension of outputs.'
    )
    parser.add_argument(
        '--num_threads',
        type=int,
        default=1,
        help='The num of threads to read tfrecords files.'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=1,
        help='The num of epochs to read tfrecords files.'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/tfrecords/',
        help='Directory of train, val and test data.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    pp.pprint(FLAGS.__dict__)
    sys.stdout.flush()
    main()
