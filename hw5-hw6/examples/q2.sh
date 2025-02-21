#!/bin/sh

./zero_order.sh f1 0.05 200 1 1 2 > q2/Z1
./zero_order.sh f1 0.05 200 2 1 2 > q2/Z2
./zero_order.sh f1 0.05 200 3 1 2 > q2/Z3

./zero_order.sh f1 0.05 200 1 1 2.03 > q2/Z4
./zero_order.sh f1 0.05 200 2 1 2.03 > q2/Z5
./zero_order.sh f1 0.05 200 3 1 2.03 > q2/Z6

./grad_desc.sh f1 0.05 200 1 2  > q2/Z7
./grad_desc.sh f1 0.05 200 1 2.03 > q2/Z8

./zero_order.sh f1 0.05 200 1 100 20 > q2/Z9
./zero_order.sh f1 0.05 200 2 100 20 > q2/Z10
./grad_desc.sh  f1 0.05 200 100 20 > q2/Z11
















