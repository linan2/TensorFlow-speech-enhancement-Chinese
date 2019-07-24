#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017    Ke Wang

"""Get train and validation set."""

from __future__ import absolute_import
from __future__ import print_function

import argparse
import os
import pprint
import random
import sys


def main():
    inputs_scp = os.path.join(FLAGS.data_dir, "inputs2.txt")
    tr_dir = os.path.join(FLAGS.data_dir, "tr")
    cv_dir = os.path.join(FLAGS.data_dir, "cv")
    tr_inputs_scp = os.path.join(tr_dir, "inputs.txt")
    cv_inputs_scp = os.path.join(cv_dir, "inputs.txt")

    print("Split to %s and %s" % (tr_dir, cv_dir))

    if not os.path.exists(tr_dir):
        os.makedirs(tr_dir)
    if not os.path.exists(cv_dir):
        os.makedirs(cv_dir)

    with open(inputs_scp, 'r') as fr_inputs, \
            open(tr_inputs_scp, 'w') as fw_tr_inputs, \
            open(cv_inputs_scp, 'w') as fw_cv_inputs:
        lists_inputs = fr_inputs.readlines()
        if len(lists_inputs) <= FLAGS.val_size:
            print(("Validation size %s is bigger than inputs scp length %s."
                   " Please reduce validation size.") % (
                       FLAGS.val_size, len(lists_inputs)))

        lists = range(len(lists_inputs))
        random.shuffle(lists)
        # print(lists)
        for i in lists:
            line_input = lists_inputs[i]
            print(line_input)
            if i < FLAGS.val_size:
                fw_cv_inputs.write(line_input)
            else:
                fw_tr_inputs.write(line_input)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help="Directory name of data to spliting."
             "(Note: inputs.scp and labels.scp)"
    )
    parser.add_argument(
        '--val_size',
        type=int,
        default=361,
        help="Validation set size."
    )

    FLAGS, unparsed = parser.parse_known_args()
    # pp = pprint.PrettyPrinter()
    # pp.pprint(FLAGS.__dict__)
    main()
