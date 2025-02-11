import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import time

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--info', type=str, required=True, help='course info')
arg_parser.add_argument('--conflicts', type=str, required=True, help='course conflicts')
arg_parser.add_argument('--quarter', type=str, required=True, help='courses this quarter')
arg_parser.add_argument('--prefs', type=str, required=False, help='instructor preferences')
arg_parser.add_argument('--config', type=str, required=True, help='configuration file')
args = arg_parser.parse_args()

class CourseScheduler:
    def __init__(self) -> None:
        pass
    