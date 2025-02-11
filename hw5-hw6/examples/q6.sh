#!/bin/sh

/nopt/python-3.6/bin/python3.6 grad_desc.py g3 0.01 200 100 > L1
/nopt/python-3.6/bin/python3.6 grad_desc.py g3 0.1 200 100 > L2

/nopt/python-3.6/bin/python3.6 grad_desc.py g3 0.3 200 100 > L3
/nopt/python-3.6/bin/python3.6 grad_desc.py g3 0.6 200 100 > L4

/nopt/python-3.6/bin/python3.6 grad_desc.py g3 1.0 200 100 > L5
/nopt/python-3.6/bin/python3.6 grad_desc.py g3 1.01 200 100 > L6











