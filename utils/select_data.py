#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017    Ke Wang

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Error parameter numbers.")
        print("Usage: python select_data.py infile1(key) ", end='')
        print("infile2(text_raw) outfile(text)")
        sys.exit(1)
    file_key = open(sys.argv[1], 'r')
    file_raw = open(sys.argv[2], 'r')
    file_text = open(sys.argv[3], 'w')

    key = []

    key_lines = file_key.readlines()
    for line in key_lines:
        line = line.decode('utf-8').strip()
        key.append(line)

    raw_lines = file_raw.readlines()
    line_num = 0
    line_total = len(key)
    for line in raw_lines:
        line_back = line.decode('utf-8').strip()
        line = line_back.split()
        if line[0] == key[line_num]:
            file_text.write(line_back.encode('utf-8'))
            file_text.write('\n')
            line_num += 1
            if line_num >= line_total:
                break
