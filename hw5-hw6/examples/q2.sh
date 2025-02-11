#!/bin/sh

/nopt/python-3.6/bin/python3.6 zero_order.py f1 0.05 200 1 1 2 > Z1
/nopt/python-3.6/bin/python3.6 zero_order.py f1 0.05 200 2 1 2 > Z2
/nopt/python-3.6/bin/python3.6 zero_order.py f1 0.05 200 3 1 2 > Z3

/nopt/python-3.6/bin/python3.6 zero_order.py f1 0.05 200 1 1 2.03 > Z4
/nopt/python-3.6/bin/python3.6 zero_order.py f1 0.05 200 2 1 2.03 > Z5
/nopt/python-3.6/bin/python3.6 zero_order.py f1 0.05 200 3 1 2.03 > Z6

/nopt/python-3.6/bin/python3.6 grad_desc.py f1 0.05 200 1 2  > Z7
/nopt/python-3.6/bin/python3.6 grad_desc.py f1 0.05 200 1 2.03 > Z8

/nopt/python-3.6/bin/python3.6 zero_order.py f1 0.05 200 1 100 20 > Z9
/nopt/python-3.6/bin/python3.6 zero_order.py f1 0.05 200 2 100 20 > Z10
/nopt/python-3.6/bin/python3.6 grad_desc.py  f1 0.05 200 100 20 > Z11
















