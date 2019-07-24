#!/bin/bash

# Copyright 2017    Ke Wang

set -euo pipefail

nj=5

echo "$0 $@"  # Print the command line for logging

. scripts/parse_options.sh || exit 1;

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
  echo "Usage: $0 [options] <data-dir> [<split-dir>]";
  echo "e.g.: $0 data/train data/train/split"
  echo "Note:  <split-dir> defaults to <data-dir>/split$nj"
  echo "Options: "
  echo "  --nj <nj>                           # number of parallel jobs"
  exit 1;
fi

inputs_scp=$1/inputs.scp
labels_scp=$1/labels.scp
if [ $# -ge 2 ]; then
  split_dir=$2
else
  split_dir=$1/split${nj}
fi

required="${inputs_scp} ${labels_scp}"
for f in $required; do
  if [ ! -f $f ]; then
    echo "$0: no such file $f"
    exit 1;
  fi
done

line_total_inputs=$(wc -l $inputs_scp | cut -d " " -f 1)
line_total_labels=$(wc -l $labels_scp | cut -d " " -f 1)
if [ ${line_total_inputs} -ne ${line_total_labels} ]; then
  echo "inputs file $1/inputs.scp and labels file $1/lables.scp are not the same line number."
  exit 1;
fi

# Split file in n lines
batch=$[line_total_inputs/nj]
if [ $[line_total_inputs%nj] -ne 0 ]; then
  batch=$[batch+1]
fi

if [ -d $split_dir ]; then
  rm -rf $split_dir || exit 1;
  mkdir -p $split_dir || exit 1;
else
  mkdir -p $split_dir || exit 1;
fi

split -l $batch -d $inputs_scp $split_dir/scp_inputs
split -l $batch -d $labels_scp $split_dir/scp_labels

i=1
for file in $split_dir/scp_inputs*; do
  mv $file $split_dir/inputs${i}.scp
  i=$[i+1]
done
i=1
for file in $split_dir/scp_labels*; do
  mv $file $split_dir/labels${i}.scp
  i=$[i+1]
done

echo "$0: split done."
