import argparse
import numpy as np
import numdifftools as nd

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--func_name', type=str)
arg_parser.add_argument('--learning_rate', type=float)
arg_parser.add_argument('--iteration_number', type=int)
arg_parser.add_argument('--x1_val', type=float)
arg_parser.add_argument('--x2_val', type=float, required=False, default=None)
args = arg_parser.parse_args()

def f1(x: list[float]) -> float:
    return x[0]**2 + x[1]**2 + 2

def f2(x: list[float]) -> float:
    return x[0]**2 + (24 * (x[1]**2))

def f3(x: list[float]) -> float:
    return x[0]**2 + (120 * (x[1]**2))

def f4(x: list[float]) -> float:
    return x[0]**2 + (1200 * (x[1]**2))

def g1(x: np.ndarray) -> float:
    return np.sin(3 * x[0])

def g2(x: np.ndarray) -> float:
    return np.sin(3 * x[0]) + (0.1 * (x[0]**2))

def g3(x: np.ndarray) -> float:
    return (x[0]**2) + 0.2

def g4(x: np.ndarray) -> float:
    return x[0]**3

def g5(x: np.ndarray) -> float:
    return ((x[0]**4) + (x[0]**2) + (10 * x[0])) / 50

#Now I'm feeling so fly like a
def g6(x):
    term1 = max(0, (3*x[0] - 2.3)**3 + 1)**2
    term2 = max(0, (-3*x[0] + 0.7)**3 + 1)**2
    return term1 + term2


class GradDescent:

    def __init__(self, learning_rate: int, iter: int, x: list[float]) -> None:
        self.learning_rate = learning_rate
        self.iter = iter
        self.x = np.atleast_1d(x)
        self.m = 5
        self.e = 0.001

    def gradient_descent(self, func) -> str:
        output = ""
        converge = False
        m_counter = 0
        wk_1 = np.array(self.x)
        gradient = nd.Gradient(func)
        w_val = ""
        for j in range(len(wk_1)):
            w_val += f"\t{wk_1[j]:.5g}"
        output += f"{0}{w_val}\t{func(wk_1):.5g}\n"

        for i in range(1, self.iter + 1):
            grad = -1 * gradient(wk_1)
            wk = wk_1 + self.learning_rate * grad
            if np.linalg.norm(wk - wk_1) < self.e and abs(func(wk) - func(wk_1)) < self.e:
                m_counter += 1
            else:
                m_counter = 0
            w_val = ""
            for j in range(len(wk)):
                w_val += f"\t{wk[j]:.5g}"
            output += f"{i}{w_val}\t{func(wk):.5g}\n"
            wk_1 = wk
            if m_counter == self.m:
                converge = True

        if not converge:
            output += "no"
        else:
            if np.linalg.norm(wk_1) < 10**8 and func(wk_1) < 10**8:
                output += "yes"
            else:
                output += "yes-but-diverge"
        return output
    
def main():
    if (args.x2_val == None):
        x = args.x1_val
    else:
        x = [args.x1_val, args.x2_val]
    descender = GradDescent(args.learning_rate, args.iteration_number, x)
    func = globals().get(args.func_name)
    result = descender.gradient_descent(func)
    print(result)

if __name__ == "__main__":
    result = main()
