#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Checks if a set of TFRecords appear to be valid.

Specifically, this checks whether the provided record sizes are consistent and
that the file does not end in the middle of a record. It does not verify the
CRCs.
"""

import struct
import tensorflow as tf

from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("input_data_pattern", "",
                    "File glob defining for the TFRecords files.")


def main(unused_argv):
  logging.set_verbosity(tf.logging.INFO)
  logging.info(FLAGS.input_data_pattern)
  paths = gfile.Glob(FLAGS.input_data_pattern)
  logging.info("Found %s files.", len(paths))
  for path in paths:
    with gfile.Open(path, "r") as f:
      first_read = True
      while True:
        length_raw = f.read(8)
        if not length_raw and first_read:
          logging.fatal("File %s has no data.", path)
          break
        elif not length_raw:
          logging.info("File %s looks good.", path)
          break
        else:
          first_read = False
        if len(length_raw) != 8:
          logging.fatal("File ends when reading record length: " + path)
          break
        length, = struct.unpack("L", length_raw)
        # +8 to include the crc values.
        record = f.read(length + 8)
        if len(record) != length + 8:
          logging.fatal("File ends in the middle of a record: " + path)
          break


if __name__ == "__main__":
  app.run()
