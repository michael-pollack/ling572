import argparse
import numpy as np

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--func_name', type=str)
arg_parser.add_argument('--learning_rate', type=float)
arg_parser.add_argument('--iteration_number', type=int)
arg_parser.add_argument('--method_id', type=str)
arg_parser.add_argument('--x1_val', type=float)
arg_parser.add_argument('--x2_val', type=float)
args = arg_parser.parse_args()

class Classifier:
    def __init__(self) -> None:
        pass

    