#!/bin/sh

/nopt/python-3.6/bin/python3.6 grad_desc.py g1 0.01 200 -1.8 > E1
/nopt/python-3.6/bin/python3.6 grad_desc.py g1 0.01 200 1.8 > E2

/nopt/python-3.6/bin/python3.6 grad_desc.py g2 0.01 200 -1.8 > E3
/nopt/python-3.6/bin/python3.6 grad_desc.py g2 0.01 200 1.8 > E4

/nopt/python-3.6/bin/python3.6 grad_desc.py g3 0.01 200 -1.8 > E5
/nopt/python-3.6/bin/python3.6 grad_desc.py g3 0.01 200 1.8 > E6

/nopt/python-3.6/bin/python3.6 grad_desc.py g4 0.01 200 -1.8 > E7
/nopt/python-3.6/bin/python3.6 grad_desc.py g4 0.01 200 1.8 > E8

/nopt/python-3.6/bin/python3.6 grad_desc.py g5 0.01 200 -1.8 > E9
/nopt/python-3.6/bin/python3.6 grad_desc.py g5 0.01 200 1.8 > E10

/nopt/python-3.6/bin/python3.6 grad_desc.py g6 0.01 200 -1.8 > E11
/nopt/python-3.6/bin/python3.6 grad_desc.py g6 0.01 200 0.77 > E12
/nopt/python-3.6/bin/python3.6 grad_desc.py g6 0.01 200 0.05 > E13
/nopt/python-3.6/bin/python3.6 grad_desc.py g6 0.01 200 0.3 > E14
/nopt/python-3.6/bin/python3.6 grad_desc.py g6 0.01 200 0.76 > E15

/nopt/python-3.6/bin/python3.6 grad_desc.py f1 0.01 200 1 2 > E16
/nopt/python-3.6/bin/python3.6 grad_desc.py f2 0.01 200 1 2 > E17
/nopt/python-3.6/bin/python3.6 grad_desc.py f3 0.01 200 1 2 > E18
/nopt/python-3.6/bin/python3.6 grad_desc.py f4 0.01 200 1 2 > E19
