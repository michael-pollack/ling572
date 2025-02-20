#!/bin/sh

#source ~/miniconda3/bin/activate "/mnt/dropbox/24-25/572/envs/570" # example: "/mnt/dropbox/23-24/WIN571/envs/571"
python3 grad_desc.py --func_name $1 --learning_rate $2 --iteration_number $3 --x1_val $4 ${5:+--x2_val "$5"}