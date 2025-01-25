#!/bin/sh

# Read the inputs
training_data=$1
test_data=$2
max_depth=$3
min_gain=$4
model_file=$5
out_path=$6

source ~/miniconda3/bin/activate "/mnt/dropbox/24-25/572/envs/570" # example: "/mnt/dropbox/23-24/WIN571/envs/571"
python build_dt.py --train $training_data --test $test_data --max_dept $max_depth --min_gain $min_gain --model $model_file --output $out_path