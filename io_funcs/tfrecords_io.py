#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017    Ke Wang     Xiaomi


"""Utility functions for working with tf.train.SequenceExamples."""

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


def get_padded_batch(file_list, batch_size, input_size, output_size,
                     left, right, num_enqueuing_threads=4,
                     num_epochs=1, infer=False):
    """Reads batches of SequenceExamples from TFRecords and pads them.
    Can deal with variable length SequenceExamples by padding each batch to the
    length of the longest sequence with zeros.
    Args:
        file_list: A list of paths to TFRecord files containing SequenceExamples.
        batch_size: The number of SequenceExamples to include in each batch.
        input_size: The size of each input vector. The returned batch of inputs
            will have a shape [batch_size, num_steps, input_size].
        output_size: The size of each output vector.
        left: An integer indicates left context number.
        right: An integer indicates right context number.
        num_enqueuing_threads: The number of threads to use for enqueuing
            SequenceExamples.
    Returns:
        utt_id: A string of inputs and labels id.
        inputs: A tensor of shape [batch_size, num_steps, input_size] of floats32s.
        labels: A tensor of shape [batch_size, num_steps] of float32s.
        lengths: A tensor of shape [batch_size] of int32s. The lengths of each
            SequenceExample before padding.
    """
    file_queue = tf.train.string_input_producer(
        file_list, num_epochs=num_epochs, shuffle=(not infer))
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_queue)

    context_features = {
        'utt_id': tf.FixedLenFeature([], tf.string),
    }

    if not infer:
        sequence_features = {
            'inputs': tf.FixedLenSequenceFeature(shape=[input_size],
                                                 dtype=tf.float32),
            'labels': tf.FixedLenSequenceFeature(shape=[output_size],
                                                 dtype=tf.float32)}

        context, sequence = tf.parse_single_sequence_example(
                serialized_example,
                context_features=context_features,
                sequence_features=sequence_features)

        length = tf.shape(sequence['inputs'])[0]

        capacity = 1000 + (num_enqueuing_threads + 1) * batch_size
        queue = tf.PaddingFIFOQueue(
            capacity=capacity,
            dtypes=[tf.string, tf.float32, tf.float32, tf.int32],
            shapes=[(), (None, input_size*(left+1+right)),
                    (None, output_size), ()])

        sfeats = splice_feats(sequence['inputs'], left, right)

        enqueue_ops = [queue.enqueue([context['utt_id'],
                                      sfeats,
                                      sequence['labels'],
                                      length])] * num_enqueuing_threads
    else:
        assert batch_size == 1, num_epochs == 1
        sequence_features = {
            'inputs': tf.FixedLenSequenceFeature(shape=[input_size],
                                                 dtype=tf.float32)}

        context, sequence = tf.parse_single_sequence_example(
            serialized_example,
            context_features=context_features,
            sequence_features=sequence_features)

        length = tf.shape(sequence['inputs'])[0]

        capacity = 1000 + (num_enqueuing_threads + 1) * batch_size
        queue = tf.PaddingFIFOQueue(
            capacity=capacity,
            dtypes=[tf.string, tf.float32, tf.int32],
            shapes=[(), (None, input_size*(left+1+right)), ()])

        sfeats = splice_feats(sequence['inputs'], left, right)

        enqueue_ops = [queue.enqueue([context['utt_id'],
                                      sfeats,
                                      length])] * num_enqueuing_threads

    tf.train.add_queue_runner(tf.train.QueueRunner(queue, enqueue_ops))
    return queue.dequeue_many(batch_size)
    print('queue dequeue_many is:',queue.dequeue_many(batch_size))
    print queue.dequeue_many(batch_size)

def read_tfrecords(file_list, input_size, output_size, num_epochs, infer=False):
    """Reads data from TFRecords.
    Args:
        file_list: A list of paths to TFRecord files containing SequenceExamples.
        input_size: The dim of each input vector.
        output_size: The dim of each output vector.
    Returns:
        A tf.train.SequenceExample containing inputs and labels(optional).
    """
    filename_queue = tf.train.string_input_producer(
        file_list, num_epochs=num_epochs, shuffle=(not infer))
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    context_features = {
        'utt_id': tf.FixedLenFeature([], tf.string),
    }

    if not infer:
        sequence_features = {
            'inputs': tf.FixedLenSequenceFeature(shape=[input_size],
                                                 dtype=tf.float32),
            'labels': tf.FixedLenSequenceFeature(shape=[output_size],
                                                 dtype=tf.float32)}

        context, sequence = tf.parse_single_sequence_example(
            serialized_example,
            context_features=context_features,
            sequence_features=sequence_features)
        return context['utt_id'], sequence['inputs'], sequence['labels']
    else:
        sequence_features = {
            'inputs': tf.FixedLenSequenceFeature(shape=[input_size],
                                                 dtype=tf.float32)}

        context, sequence = tf.parse_single_sequence_example(
            serialized_example,
            context_features=context_features,
            sequence_features=sequence_features)
        return context['utt_id'], sequence['inputs']


def splice_feats(feats, left, right):
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
    for i in range(left, 0, -1):
        fl = tf.slice(feats, [0, 0], [row-i, -1])
        for j in range(i):
            fl = tf.pad(fl, [[1, 0], [0, 0]], mode='SYMMETRIC')
        sfeats.append(fl)
    sfeats.append(feats)

    # Right
    for i in range(1, right+1):
        fr = tf.slice(feats, [i, 0], [-1, -1])
        for j in range(i):
            fr = tf.pad(fr, [[0, 1], [0, 0]], mode='SYMMETRIC')
        sfeats.append(fr)

    return tf.concat(sfeats, 1)


def get_batch(file_list, batch_size, input_size, output_size, left, right,
              num_enqueuing_threads=4, num_epochs=1,infer=False):
    """Reads batches of Examples from TFRecords and splice them.
    Args:
        file_list:  A list of paths to TFRecord files containing SequenceExamples.
        batch_size: The number of Examples to include in each batch.
        input_size: The size of each input vector. The returned batch of inputs
            will have a shape [batch_size, input_size*(left+1+right)].
        num_enqueuing_threads: The number of threads to use for enqueuing Examples.
    Return:
        batch_x: A tensor of shape [batch_size, input_size*(left+1+right)] of floats32s.
        batch_y: A tensor of shape [batch_size, output_size] of float32s.
    """
    if not infer:
        utt_id, inputs, labels = read_tfrecords(file_list, input_size,
                                                output_size, num_epochs, infer)
    else:
        utt_id, inputs = read_tfrecords(file_list, input_size, output_size,
                                        num_epochs, infer)
    sfeats = splice_feats(inputs, left, right)

    if infer:
        assert batch_size == 1, num_epochs == 1

    length = tf.shape(inputs)[0]
    capacity = 1000 + (num_enqueuing_threads + 1) * batch_size
    if not infer:
        slice_queue = tf.RandomShuffleQueue(
            capacity=capacity,
            min_after_dequeue = 1000,
            dtypes=[tf.float32, tf.float32],
            shapes=[[input_size*(left+1+right),], [output_size,]])
        enqueue_ops = [slice_queue.enqueue_many(
            [sfeats, labels])] * num_enqueuing_threads
    else:
        slice_queue = tf.PaddingFIFOQueue(
            capacity=capacity,
            dtypes=[tf.string, tf.float32, tf.int32],
            shapes=[(), (None, input_size*(left+1+right)), ()])
        enqueue_ops = [slice_queue.enqueue(
            [utt_id, sfeats, length])] * num_enqueuing_threads

    tf.train.add_queue_runner(tf.train.QueueRunner(slice_queue, enqueue_ops))

    if not infer:
        batch_x, batch_y = slice_queue.dequeue_many(batch_size)
        return batch_x, batch_y
    else:
        batch_id, batch_x, batch_len = slice_queue.dequeue_many(batch_size)
        return batch_id, batch_x, batch_len
