#!/bin/sh

./grad_desc.sh g1 0.01 200 -1.8 > q4/E1
./grad_desc.sh g1 0.01 200 1.8 > q4/E2

./grad_desc.sh g2 0.01 200 -1.8 > q4/E3
./grad_desc.sh g2 0.01 200 1.8 > q4/E4

./grad_desc.sh g3 0.01 200 -1.8 > q4/E5
./grad_desc.sh g3 0.01 200 1.8 > q4/E6
./grad_desc.sh g4 0.01 200 -1.8 > q4/E7
./grad_desc.sh g4 0.01 200 1.8 > q4/E8

./grad_desc.sh g5 0.01 200 -1.8 > q4/E9
./grad_desc.sh g5 0.01 200 1.8 > q4/E10

./grad_desc.sh g6 0.01 200 -1.8 > q4/E11
./grad_desc.sh g6 0.01 200 0.77 > q4/E12
./grad_desc.sh g6 0.01 200 0.05 > q4/E13
./grad_desc.sh g6 0.01 200 0.3 > q4/E14
./grad_desc.sh g6 0.01 200 0.76 > q4/E15

./grad_desc.sh f1 0.01 200 1 2 > q4/E16
./grad_desc.sh f2 0.01 200 1 2 > q4/E17
./grad_desc.sh f3 0.01 200 1 2 > q4/E18
./grad_desc.sh f4 0.01 200 1 2 > q4/E19
