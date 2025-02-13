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
    
class Minimizer:

    def __init__(self, learning_rate: int, iter: int, method: str, x1: float, x2: float, func) -> float:
        self.learning_rate = learning_rate
        self.iter = iter
        self.x1 = x1
        self.x2 = x2
        if method == '1':
            score = self.random_search(func)
        elif method == '2':
            score = self.coordinate_search(func)
        else:
            score = self.coordinate_descent(func)
        return score
        

    def random_search(self, func) -> float:
        best_x1, best_x2 = None, None
        best_score = float("inf")

        for _ in range(10):
            x1 = np.random.uniform(-1, 1)
            x2 = math.sqrt(1 - (x1**2))
            score = func(x1, x2)

            if score < best_score:
                best_score = score
                best_x1, best_x2 = x1, x2

        return best_x1, best_x2, best_score

    def coordinate_search(self, func) -> float:
        x1, x2 = self.x1, self.x2
        best_score = func(x1, x2)

        for _ in range(self.iter):
            improved = False
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

            for dx1, dx2 in directions:
                new_x1, new_x2 = x1 + dx1, x2 + dx2
                new_score = func(new_x1, new_x2)

                if new_score < best_score:
                    best_score = new_score
                    x1, x2 = new_x1, new_x2
                    improved

            if not improved:
                break
        
        return x1, x2, best_score

    def coordinate_descent(self, func) -> float:
        x1, x2 = self.x1, self.x2
        best_score = func(x1, x2)

        for i in range(self.iter):
            improved = False

            if i % 2 == 1:
                for dx1 in [self.learning_rate, -self.learning_rate]:
                    new_x1 = x1 + dx1
                    new_score = func(new_x1)
                    if new_score < best_score:
                        x1 = new_x1
                        best_score = new_score
                        improved = True
            
            else: 
                for dx2 in [self.learning_rate, -self.learning_rate]:
                    new_x2 = x2 + dx2
                    new_score = func(new_x2)
                    if new_score < best_score:
                        x2 = new_x2
                        best_score = new_score
                        improved = True
            
            if not improved:
                break
        
        return x1, x2, best_score








