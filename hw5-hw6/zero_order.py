import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import math
import time

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--train', type=str, required=True, help='training data')
arg_parser.add_argument('--test', type=str, required=True, help='test data')
arg_parser.add_argument('--k', type=str, required=True, help='k value')
arg_parser.add_argument('--function', type=str, required=False, help='similarity function')
arg_parser.add_argument('--output', type=str, required=True, help='output file')
args = arg_parser.parse_args()

def f1(x1: float, x2: float) -> float:
    return x1**2 + x2**2 + 2

def f2(x1: float, x2: float) -> float:
    return x1**2 + (24 * (x2**2))

def f3(x1: float, x2: float) -> float:
    return x1**2 + (120 * (x2**2))

def f4(x1: float, x2: float) -> float:
    return x1**2 + (1200 * (x2**2))

def g1(x: float) -> float:
    return math.sin(3 * x)

def g2(x: float) -> float:
    return math.sin(3 * x) + (0.1 * (x**2))

def g3(x: float) -> float:
    return (x**2) + 0.2

def g4(x: float) -> float:
    return x**3

def g5(x: float) -> float:
    return ((x**4) + (x**2) + (10 * x)) / 50

#Now I'm feeling so fly like a
def g6(x: float) -> float:
    return max(0, (((3 * x) - 2.3)**3 + 1))**2 + max(0, (((-3 * x) - 0.7)**3 + 1))**2