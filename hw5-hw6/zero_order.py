import argparse
import numpy as np
import math

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--func_name', type=str)
arg_parser.add_argument('--learning_rate', type=float)
arg_parser.add_argument('--iteration_number', type=int)
arg_parser.add_argument('--method_id', type=str)
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
    return math.sin(3 * x[0])

def g2(x: list[float]) -> float:
    return math.sin(3 * x[0]) + (0.1 * (x[0]**2))

def g3(x: list[float]) -> float:
    return (x[0]**2) + 0.2

def g4(x: list[float]) -> float:
    return x[0]**3

def g5(x: list[float]) -> float:
    return ((x[0]**4) + (x[0]**2) + (10 * x[0])) / 50

#Now I'm feeling so fly like a
def g6(x: list[float]) -> float:
    return max(0, (((3 * x[0]) - 2.3)**3 + 1))**2 + max(0, (((-3 * x[0]) - 0.7)**3 + 1))**2
    
class Minimizer:

    def __init__(self, learning_rate: int, iter: int, method: str, x1: float, x2: float, func: str) -> None:
        print("Hiya")
        self.learning_rate = learning_rate
        self.iter = iter
        self.x1 = x1
        self.x2 = x2
        local_func = globals().get(func)
        if method == '1':
            self.output = self.random_search(local_func)
        elif method == '2':
            self.output = self.coordinate_search(local_func)
        else:
            self.output = self.coordinate_descent(local_func)

        

    def random_search(self, func) -> str:
        best_score = float("inf")
        output = ""
        for i in range(self.iter):
            for _ in range(10):
                rand_x1 = np.random.uniform(-1, 1)
                rand_x2 = np.sqrt(1 - (x1**2))
                directions = [(rand_x1, rand_x2), (rand_x1, -1*rand_x2)]

                for dx1, dx2 in directions:
                    new_x1 = x1 + self.learning_rate * dx1
                    new_x2 = x2 + self.learning_rate * dx2
                    new_score = func([new_x1, new_x2])

                if new_score < best_score:
                    best_score = new_score
                    x1, x2 = new_x1, new_x2

            output += f"{i}\t{x1}\t{x2}\t{best_score}\n"

        return output

    def coordinate_search(self, func) -> str:
        x1, x2 = self.x1, self.x2
        best_score = func([x1, x2])
        output = ""

        for i in range(self.iter):
            improved = False
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

            for dx1, dx2 in directions:
                new_x1, new_x2 = x1 + (dx1 * self.learning_rate), x2 + (dx2 * self.learning_rate)
                new_score = func([new_x1, new_x2])

                if new_score < best_score:
                    best_score = new_score
                    x1, x2 = new_x1, new_x2
                    improved = True
            
            output += f"{i}\t{x1}\t{x2}\t{best_score}\n"
            if not improved:
                break
        
        return output

    def coordinate_descent(self, func) -> str:
        x1, x2 = self.x1, self.x2
        best_score = func([x1, x2])
        improved = 0
        output = ""

        for i in range(self.iter):

            if i % 2 == 1:
                for dx1 in [self.learning_rate, -self.learning_rate]:
                    new_x1 = x1 + (dx1 * self.learning_rate)
                    new_score = func(new_x1)
                    if new_score < best_score:
                        x1 = new_x1
                        best_score = new_score
                        improved = 0
                    else:
                        improved += 1
            
            else: 
                for dx2 in [self.learning_rate, -self.learning_rate]:
                    new_x2 = x2 + (dx2 * self.learning_rate)
                    new_score = func(new_x2)
                    if new_score < best_score:
                        x2 = new_x2
                        best_score = new_score
                        improved = 0
                    else:
                        improved += 1
            
            output += f"{i}\t{x1}\t{x2}\t{best_score}\n"
            if improved >= 2:
                break
        
        return output

def main():
    result = Minimizer(args.learning_rate, args.iteration_number, args.method_id, args.x1_val, args.x2_val, args.func_name)
    print(result.output)

    
if __name__ == "__main__":
    result = main()






 