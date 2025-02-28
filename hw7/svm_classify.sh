#!/bin/sh

source ~/miniconda3/bin/activate "/mnt/dropbox/24-25/572/envs/570" # example: "/mnt/dropbox/23-24/WIN571/envs/571"
python3 svm_classify.py --test_data $1 --model_file $2 --sys_output $3
