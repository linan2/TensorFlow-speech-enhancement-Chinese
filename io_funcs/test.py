# -*- coding: utf-8 -*-

from kaldi_io import ArkReader
import numpy as np
import random

inputs_path = '/home/train02/linan/ASR/Ganspeechenhan/rsrgan/data/tr/inputs.ark'
inputs_offset = '836825'
reader = ArkReader()
#a = {"a","b","c","d","e"}
inputs_path2 = '/home/train02/linan/ASR/Ganspeechenhan/rsrgan/data/simuda/labels.ark'
inputs_offset2 = '836829'

inputs2 = reader.read_ark(inputs_path2, inputs_offset2).astype(np.float64)
inputs = reader.read_ark(inputs_path, inputs_offset).astype(np.float64)
print inputs2.shape
print inputs
print inputs.shape
#lists = range(len(a))
#random.shuffle(lists)
#for i in xrange(len(lists)):
#    print i

