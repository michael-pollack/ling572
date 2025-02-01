#!/bin/sh

# Read the inputs
training_data=$1
test_data=$2
cpdelta=$3
prdelta=$4
model=$5
output=$6


source ~/miniconda3/bin/activate "/mnt/dropbox/24-25/572/envs/570" # example: "/mnt/dropbox/23-24/WIN571/envs/571"
python build_nb.py --train $training_data --test $test_data --model $model --output $output --cpdelta $cpdelta --prdelta $prdelta