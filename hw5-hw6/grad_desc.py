import argparse
import numpy as np
import numdifftools as nd

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--func_name', type=str)
arg_parser.add_argument('--learning_rate', type=float)
arg_parser.add_argument('--iteration_number', type=int)
arg_parser.add_argument('--x1_val', type=float)
arg_parser.add_argument('--x2_val', type=float)
args = arg_parser.parse_args()

def f1(x: list[float]) -> float:
    return x[0]**2 + x[1]**2 + 2

def f2(x: list[float]) -> float:
    return x[0]**2 + (24 * (x[1]**2))

def f3(x: list[float]) -> float:
    return x[0]**2 + (120 * (x[1]**2))

def f4(x: list[float]) -> float:
    return x[0]**2 + (1200 * (x[1]**2))

def g1(x: list[float]) -> float:
    return np.sin(3 * x[0])

def g2(x: list[float]) -> float:
    return np.sin(3 * x[0]) + (0.1 * (x[0]**2))

def g3(x: list[float]) -> float:
    return (x[0]**2) + 0.2

def g4(x: list[float]) -> float:
    return x[0]**3

def g5(x: list[float]) -> float:
    return ((x[0]**4) + (x[0]**2) + (10 * x[0])) / 50

#Now I'm feeling so fly like a
def g6(x: list[float]) -> float:
    return max(0, (((3 * x[0]) - 2.3)**3 + 1))**2 + max(0, (((-3 * x[0]) - 0.7)**3 + 1))**2

class GradDescent:

    def __init__(self, learning_rate: int, iter: int, x: list[float]) -> None:
        self.learning_rate = learning_rate
        self.iter = iter
        self.x = x
        self.m = 5
        self.e = 0.001

    def gradient_descent(self, func) -> str:
        output = ""
        x = np.array(self.x)
        history = []
        gradient = nd.Gradient(func)

        for i in range(self.iter):
            grad = gradient(x)
            x = x - self.learning_rate * grad
            func_output = func(x)

        x_val = ""
        for j in len(x):
            x_val += f"\t{x[j]}"
        
        output += f"{i}{x_val}\t{func_output}\n"
        return output
    
def main():
    if (args.x2_val == None):
        x = args.x1_val
    else:
        x = [args.x1_val, args.x2_val]
    descender = GradDescent(args.learning_rate, args.iteration_number, x)
    func = globals.get(args.func_name)
    result = descender.gradient_descent(func)
    print(result)


