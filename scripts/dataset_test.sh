#!/bin/bash

# Copyright 2017    Ke Wang

set -euo pipefail

stage=2

train_dir=data/train/io_test
#train_dir=data/train/train_100h

if [ $stage -le 0 ]; then
  python io_funcs/convert_cmvn_to_numpy.py \
    --inputs=$train_dir/inputs.cmvn \
    --labels=$train_dir/labels.cmvn \
    --save_dir=$train_dir
fi

if [ $stage -le 1 ]; then
  nj=1
  logdir=exp/
  if [ ! -d $logdir ]; then
    mkdir -p $logdir
  fi

  rm -rf $logdir/.error || exit 1;
  bash scripts/split_scp.sh --nj $nj $train_dir
  for i in $(seq $nj); do
  (
    python io_funcs/make_tfrecords.py \
      --inputs=$train_dir/split${nj}/inputs${i}.scp \
      --labels=$train_dir/split${nj}/labels${i}.scp \
      --cmvn_dir=$train_dir \
      --apply_cmvn=true \
      --output_dir=$train_dir/tfrecords \
      --name="train${i}"
  ) || touch $logdir/.error &
  done
  wait
  [ -f $logdir/.error ] && \
    echo "$0: there was a problem while making TFRecords" && exit 1
  echo "Making TFRecords done."
fi

if [ $stage -le 2 ]; then
  CUDA_VISIBLE_DEVICES="3" python io_funcs/tfrecords_dataset_test.py \
    --batch_size=128 \
    --input_dim=257 \
    --output_dim=40 \
    --num_threads=32 \
    --num_epochs=1 \
    --data_dir=$train_dir/tfrecords
fi

