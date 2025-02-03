#!/bin/sh

# Read the inputs
training_data=$1
test_data=$2
k=$3
function=$4
output=$5


#source ~/miniconda3/bin/activate "/mnt/dropbox/24-25/572/envs/570" # example: "/mnt/dropbox/23-24/WIN571/envs/571"
python build_knn.py --train $training_data --test $test_data --k $k --function $function --output $output 