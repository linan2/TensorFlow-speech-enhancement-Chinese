#!/bin/bash

# Copyright 2019.7    Nan LEE

set -euo pipefail

stage=0

nj=1
val_size=500
train_dir=data
test_dir=data/test
logdir=exp
tr_list=$train_dir/tr.list
cv_list=$train_dir/cv.list
test_list=$test_dir/test.list
save_dir=exp/dnn

# Data prepare
if [ $stage -le 0 ]; then
  echo "Prepare tr and cv data"
  python scripts/get_train_val_scp.py --data_dir=$train_dir --val_size 500 
  date
  # Make TFRecords file
  echo "Begin making TFRecords files ..."
  if [ ! -d $logdir ]; then
    mkdir -p $logdir || exit 1;
  fi

  # cv set
  declare -i verbose=30
  [ -d $train_dir/tfrecords ] && (rm -rf $train_dir/tfrecords || exit 1;)
  mkdir -p $train_dir/tfrecords || exit 1;
  
  TF_CPP_MIN_LOG_LEVEL=1 python io_funcs/make_setf.py --inputs=$train_dir/cv/inputs_feat.txt --name="cv"
  echo "$train_dir/tfrecords/cv.tfrecords" > $cv_list
 wait
 date

 TF_CPP_MIN_LOG_LEVEL=1 python io_funcs/make_setf.py --inputs=$train_dir/tr/inputs_feat.txt --name="tr"
 echo "$train_dir/tfrecords/tr.tfrecords" > $tr_list
  wait
  date

  [ -f $train_dir/batch_num.txt ] && rm $train_dir/batch_num.txt
  echo "Make train TFRecords files sucessed."
  echo ""
fi
#exit 0;
# Train model
if [ $stage -le 2 ]; then
  echo "$(date): $(hostname)"
 CUDA_VISIBLE_DEVICES="1,2,3" TF_CPP_MIN_LOG_LEVEL=2 \
    python scripts/train_dnn.py \
      --data_dir=$train_dir \
      --tr_list_file=$tr_list \
      --cv_list_file=$cv_list \
      --g_type="res_rced" \
      --save_dir=$save_dir \
      --batch_size=64 \
      --g_learning_rate=0.001 \
      --keep_lr=2 \
      --batch_norm=true \
      --keep_prob=1 \
      --l2_scale=0 \
      --input_dim=257 \
      --output_dim=257 \
      --left_context=5 \
      --right_context=5 \
      --min_epoches=10 \
      --max_epoches=12 \
      --decay_factor=0.8 \
      --start_halving_impr=0.01 \
      --end_halving_impr=0.001 \
      --num_threads=1 \
      --num_gpu=1 || exit 1;

  echo "Finished training successfully on $(date)"
  echo ""
fi
# exit 0;

# Decode

if [ $stage -le 4 ]; then
  echo "Prepare test data"
  if [ -f $logdir/.test.error ]; then
    rm -rf $logdir/.test.error || exit 1;
  fi
  declare -i verbose=30
  # [ -d $test_dir/tfrecords ] && (rm -rf $test_dir/tfrecords || exit 1;)
  # mkdir -p $test_dir/tfrecords || exit 1;
 for datase in data/test/*;do
 # for datase in data/simusi;do
  rm -rf $datase/tfrecords 
  TF_CPP_MIN_LOG_LEVEL=1 python io_funcs/make_sete.py \
    --inputs=$datase/inputs.txt \
    --output_dir=$datase/tfrecords \
    --name="test" || touch $logdir/.test.error &
  echo "$datase/tfrecords/test.tfrecords" > $datase/test.list
  # exit 0;
  wait
 done
fi
# Decode
if [ $stage -le 5 ]; then

  echo "Start decoding test data"
for datase in data/test/*;do
# for datase in data/simusi;do
    CUDA_VISIBLE_DEVICES="1" TF_CPP_MIN_LOG_LEVEL=2 python scripts/train_dnn.py \
      --decode \
      --data_dir=$train_dir \
      --test_list_file=$datase/test.list \
      --g_type="res_rced" \
      --save_dir=$save_dir \
	  --g_learning_rate=0.001 \
      --batch_norm=true \
      --input_dim=257 \
      --output_dim=257 \
      --left_context=5 \
      --right_context=5 \
      --batch_size=1 \
      --keep_prob=1 \
      --l2_scale=0 \
      --num_threads=1 \
	  --savetestdir=$datase || exit 1;
  echo "Decoding done"
wait
done
fi

exit 0
