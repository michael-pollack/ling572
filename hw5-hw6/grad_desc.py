import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import math
import time

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--func', type=str, required=True, help='training data')
arg_parser.add_argument('--rate', type=str, required=True, help='test data')
arg_parser.add_argument('--iter', type=str, required=True, help='k value')
arg_parser.add_argument('--method', type=str, required=False, help='similarity function')
arg_parser.add_argument('--x1', type=str, required=True, help='output file')
arg_parser.add_argument('--x2', type=str, required=True, help='output file')
args = arg_parser.parse_args()

class GradDescent:
    def __init__(self) -> None:
        pass