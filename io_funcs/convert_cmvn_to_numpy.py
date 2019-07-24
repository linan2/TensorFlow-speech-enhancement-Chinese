#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017    Ke Wang

"""Convert inputs and lables GLOBAL cmvns to a Numpy file."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import struct

import numpy as np

def convert_cmvn_to_numpy(inputs_cmvn, labels_cmvn, save_dir):
    """Convert global binary ark cmvn to numpy format."""

    print("Convert %s and %s to Numpy format" % (inputs_cmvn, labels_cmvn))
    inputs_filename = inputs_cmvn
    labels_filename = labels_cmvn

    inputs = read_binary_file(inputs_filename, 0)
    labels = read_binary_file(labels_filename, 0)

    inputs_frame = inputs[0][-1]
    labels_frame = labels[0][-1]

    # assert inputs_frame == labels_frame

    cmvn_inputs = np.hsplit(inputs, [inputs.shape[1] - 1])[0]
    cmvn_labels = np.hsplit(labels, [labels.shape[1] - 1])[0]

    mean_inputs = cmvn_inputs[0] / inputs_frame
    stddev_inputs = np.sqrt(cmvn_inputs[1] / inputs_frame - mean_inputs ** 2)
    mean_labels = cmvn_labels[0] / labels_frame
    stddev_labels = np.sqrt(cmvn_labels[1] / labels_frame - mean_labels ** 2)

    cmvn_name = os.path.join(save_dir, "train_cmvn.npz")
    np.savez(cmvn_name,
             mean_inputs=mean_inputs,
             stddev_inputs=stddev_inputs,
             mean_labels=mean_labels,
             stddev_labels=stddev_labels)

    print("Write to %s" % cmvn_name)


def read_binary_file(filename, offset=0):
    """Read data from matlab binary file (row, col and matrix).

    Returns:
        A numpy matrix containing data of the given binary file.
    """
    read_buffer = open(filename, 'rb')
    read_buffer.seek(int(offset), 0)
    header = struct.unpack('<xcccc', read_buffer.read(5))
    if header[0] != 'B':
        print("Input .ark file is not binary")
        sys.exit(-1)
    if header[1] == 'C':
        print("Input .ark file is compressed, exist now.")
        sys.exit(-1)

    rows = 0; cols= 0
    _, rows = struct.unpack('<bi', read_buffer.read(5))
    _, cols = struct.unpack('<bi', read_buffer.read(5))

    if header[1] == "F":
        tmp_mat = np.frombuffer(read_buffer.read(rows * cols * 4),
                                dtype=np.float32)
    elif header[1] == "D":
        tmp_mat = np.frombuffer(read_buffer.read(rows * cols * 8),
                                dtype=np.float64)
    mat = np.reshape(tmp_mat, (rows, cols))
    read_buffer.close()

    return mat


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--inputs',
        type=str,
        default='data/train/inputs.cmvn',
        help="Name of input CMVN file."
    )

    parser.add_argument(
        '--labels',
        type=str,
        default='data/train/labels.cmvn',
        help="Name of label CMVN file."
    )
    parser.add_argument(
        '--save_dir',
        required=True,
        help="Directory to save Numpy format CMVN file."
    )
    FLAGS, unparsed = parser.parse_known_args()

    convert_cmvn_to_numpy(FLAGS.inputs, FLAGS.labels, FLAGS.save_dir)
