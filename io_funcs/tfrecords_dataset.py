#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017    Ke Wang     Xiaomi


"""Utility functions for import data using tf.contrib.data.Dataset.
Make sure TensorFlow vesion >= 1.2.0
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def make_sequence_example(utt_id, inputs, labels=None):
    """Returns a SequenceExample for the given inputs and labels(optional).
    Args:
        utt_id: The key of corresponding features. It is very useful for decoding.
        inputs: A list of input vectors. Each input vector is a list of floats.
        labels(optional): A list of label vectors. Each label vector is a list of floats.
    Returns:
        A tf.train.SequenceExample containing inputs and labels(optional).
    """
    utt_id_feature = {
        'utt_id': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[utt_id]))
    }
    input_features = [
        tf.train.Feature(float_list=tf.train.FloatList(value=input_))
        for input_ in inputs]

    context = tf.train.Features(feature=utt_id_feature)

    if labels is not None:
        label_features = [
            tf.train.Feature(float_list=tf.train.FloatList(value=label_))
            for label_ in labels]
        feature_list = {
            'inputs': tf.train.FeatureList(feature=input_features),
            'labels': tf.train.FeatureList(feature=label_features)
        }
    else:
        feature_list = {
            'inputs': tf.train.FeatureList(feature=input_features)
        }
    feature_lists = tf.train.FeatureLists(feature_list=feature_list)
    return tf.train.SequenceExample(context=context, feature_lists=feature_lists)


def get_padded_batch(filenames, batch_size, input_size, output_size,
                     left_context, right_context, num_threads=4,
                     num_epochs=1, num_buckets=20):
    """Reads batches of SequenceExamples from TFRecords and pads them.
    Can deal with variable length SequenceExamples by padding each batch to the
    length of the longest sequence with zeros.
    Args:
        filename: A list of paths to TFRecord files containing SequenceExamples.
        batch_size: The number of SequenceExamples to include in each batch.
        input_size: The size of each input vector. The returned batch of inputs
            will have a shape [batch_size, num_steps, input_size].
        output_size: The size of each output vector.
        left_context: An integer indicates left context number.
        right_context: An integer indicates right context number.
        num_threads: The number of threads to use for enqueuing SequenceExamples.
    Returns:
        utt_id: A string of inputs and labels id.
        inputs: A tensor of shape [batch_size, num_steps,
            input_size * (left_context + right_context + 1)] of floats32s.
        labels: A tensor of shape [batch_size, num_steps] of float32s.
        lengths: A tensor of shape [batch_size] of int32s. The lengths of each
            SequenceExample before padding.
    """
    buffer_size = 10000
    # dataset = tf.contrib.data.TFRecordDataset(filenames)
    dataset = tf.data.TFRecordDataset(filenames)
    print("done getTFRecordDataset")

    def splice_feats(feats, left_context, right_context):
        """Splice feats like KALDI.
        Args:
            feats: input feats have a shape [row, col].
            left: left context number.
            right: right context number.
        Returns:
            Spliced feats with a shape [row, col*(left+1+right)]
        """
        sfeats = []
        row = tf.shape(feats)[0]
        # Left
        for i in range(left_context, 0, -1):
            fl = tf.slice(feats, [0, 0], [row-i, -1])
            for j in range(i):
                fl = tf.pad(fl, [[1, 0], [0, 0]], mode='SYMMETRIC')
            sfeats.append(fl)
        sfeats.append(feats)

        # Right
        for i in range(1, right_context+1):
            fr = tf.slice(feats, [i, 0], [-1, -1])
            for j in range(i):
                fr = tf.pad(fr, [[0, 1], [0, 0]], mode='SYMMETRIC')
            sfeats.append(fr)
        return tf.concat(sfeats, 1)

    def parser_train(record):
        """Extract data from a `tf.SequenceExamples` protocol buffer for training."""
        context_features = {
            'utt_id': tf.FixedLenFeature([], tf.string),
        }
        sequence_features = {
            'inputs': tf.FixedLenSequenceFeature(shape=[input_size],
                                                 dtype=tf.float32),
            'labels': tf.FixedLenSequenceFeature(shape=[output_size],
                                                 dtype=tf.float32)}

        context, sequence = tf.parse_single_sequence_example(
            record,
            context_features=context_features,
            sequence_features=sequence_features)
        splice_inputs = splice_feats(sequence['inputs'], left_context, right_context)
        return context['utt_id'], splice_inputs, sequence['labels']

    input_dim = input_size * (left_context + 1 + right_context)
    dataset = dataset.map(parser_train,
                          num_parallel_calls=num_threads)
                          # num_threads=num_threads,
                          # output_buffer_size=buffer_size)
    # Add in sequence lengths.
    dataset = dataset.map(lambda key, src, tgt: (key, src, tgt, tf.shape(src)[0]),
                          num_parallel_calls=num_threads)
                          # num_threads=num_threads,
                          # output_buffer_size=buffer_size)
    dataset.prefetch(buffer_size)
    # Shuffle
    dataset = dataset.shuffle(buffer_size=buffer_size)
    # Bucket by source sequence length.
    def batching_func(x):
        return x.padded_batch(
            batch_size,
            # The first two entries are the source and target feature;
            # these are unknown-row-number matrix.  The last entry is
            # the source and target row size; this is scalar.
            padded_shapes=(tf.TensorShape([]),                  # key
                           tf.TensorShape([None, input_dim]),   # src
                           tf.TensorShape([None, output_size]), # tgt
                           tf.TensorShape([])),                 # src_len
            padding_values=('',    # key -- unused
                            0.0,   # src
                            0.0,   # tgt
                            0))    # src_len -- unused

    # We recommend calling repeat before batch
    dataset = dataset.repeat(num_epochs)
    # Batch
    if num_buckets > 1:
        def key_func(unused_1, unused_2, unused_3, src_len):
            # Calculate bucket_width by maximum source sequence length.
            # Pairs with length [0, 200 + bucket_width) go to bucket 0, length
            # [200 + bucket_width, 200 + 2 * bucket_width) go to bucket 1, etc.
            # Pairs with length over ((num_bucket - 1) * bucket_width + 200)
            # words all go into the last bucket.
            start_width = 200
            bucket_width = 50
            bucket_id = (src_len - start_width) // bucket_width
            return tf.to_int64(tf.minimum(num_buckets, bucket_id))
        def reduce_func(unused_key, windowed_data):
            return batching_func(windowed_data)
        batched_dataset = dataset.apply(tf.contrib.data.group_by_window(
            key_func=key_func, reduce_func=reduce_func, window_size=batch_size))
        # batched_dataset = dataset.group_by_window(
        #     key_func=key_func, reduce_func=reduce_func, window_size=batch_size)
    else:
        batched_dataset = batching_func(dataset)

    batched_iter = batched_dataset.make_one_shot_iterator()

    utt_id, features, labels, lengths = batched_iter.get_next()
    return utt_id, features, labels, lengths


def get_batch(filenames, batch_size, input_size, output_size,
              left_context, right_context, num_threads,
              num_epochs=1, infer=False):
    """Reads batches of Examples from TFRecords and splice them.
    Args:
        filenames:  A list of paths to TFRecord files containing SequenceExamples.
        batch_size: The number of Examples to include in each batch.
        input_size: The size of each input vector. The returned batch of inputs
            will have a shape [batch_size,
                input_size * (left_context + 1 + right_context)].
        num_threads: The number of threads to use for processing elements in
            parallel. If not specified, elements will be processed
            sequentially without buffering.
    Return:
        features: A tensor of shape [batch_size,
            input_size * (left_context + 1 + right_context)] of floats32s.
        labels: A tensor of shape [batch_size, output_size] of float32s.
    """
    buffer_size = 10000
    # dataset = tf.contrib.data.TFRecordDataset(filenames)
    dataset = tf.data.TFRecordDataset(filenames)

    def splice_feats(feats, left_context, right_context):
        """Splice feats like KALDI.
        Args:
            feats: input feats have a shape [row, col].
            left_context: left context number.
            right_context: right context number.
        Returns:
            Spliced feats with a shape [row,
                col * (left_context + 1 + right_context)]
        """
        sfeats = []
        row = tf.shape(feats)[0]
        # Left
        for i in range(left_context, 0, -1):
            fl = tf.slice(feats, [0, 0], [row-i, -1])
            for j in range(i):
                fl = tf.pad(fl, [[1, 0], [0, 0]], mode='SYMMETRIC')
            sfeats.append(fl)
        sfeats.append(feats)

        # Right
        for i in range(1, right_context+1):
            fr = tf.slice(feats, [i, 0], [-1, -1])
            for j in range(i):
                fr = tf.pad(fr, [[0, 1], [0, 0]], mode='SYMMETRIC')
            sfeats.append(fr)
        return tf.concat(sfeats, 1)

    def parser_infer(record):
        """Extract data from a `tf.SequenceExamples` protocol buffer for infering."""
        context_features = {
            'utt_id': tf.FixedLenFeature([], tf.string),
        }
        sequence_features = {
            'inputs': tf.FixedLenSequenceFeature(shape=[input_size],
                                                 dtype=tf.float32)}

        context, sequence = tf.parse_single_sequence_example(
            record,
            context_features=context_features,
            sequence_features=sequence_features)
        splice_inputs = splice_feats(sequence['inputs'], left_context, right_context)
        length = tf.shape(splice_inputs)[0]
        return context['utt_id'], splice_inputs, length

    def parser_train(record):
        """Extract data from a `tf.SequenceExamples` protocol buffer for training."""
        context_features = {
            'utt_id': tf.FixedLenFeature([], tf.string),
        }
        sequence_features = {
            'inputs': tf.FixedLenSequenceFeature(shape=[input_size],
                                                 dtype=tf.float32),
            'labels': tf.FixedLenSequenceFeature(shape=[output_size],
                                                 dtype=tf.float32)}

        context, sequence = tf.parse_single_sequence_example(
            record,
            context_features=context_features,
            sequence_features=sequence_features)
        splice_inputs = splice_feats(sequence['inputs'], left_context, right_context)
        return splice_inputs, sequence['labels']

    if not infer:
        dataset = dataset.map(parser_train,
                              num_parallel_calls=num_threads)
                              # num_threads=num_threads,
                              # output_buffer_size=buffer_size)
        dataset = dataset.unbatch()
        dataset = dataset.shuffle(buffer_size=buffer_size)
    else:
        assert batch_size == 1, num_epochs == 1
        dataset = dataset.map(parser_infer,
                              num_parallel_calls=num_threads)
                              # num_threads=num_threads,
                              # output_buffer_size=buffer_size)
    dataset.prefetch(buffer_size)
    # We recommend calling repeat before batch
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()

    # `features` is a batch of features; `labels` is a batch of labels.
    if not infer:
        features, labels = iterator.get_next()
        return features, labels
    else:
        utt_ids, features, length = iterator.get_next()
        return utt_ids, features, length

