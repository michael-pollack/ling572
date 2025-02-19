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
    
class Minimizer:

    def __init__(self, learning_rate: int, iter: int, method: str, x1: float, x2: float, func: str) -> None:
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
        x1, x2 = self.x1, self.x2
        best_score = func([x1, x2])
        output = f"{0}\t{x1:.5g}\t{x2:.5g}\t{best_score:.5g}\n"
        for i in range(1, self.iter + 1):
            directions = []
            for _ in range(10):
                dx1 = np.random.uniform(-1, 1)
                dx2 = np.sqrt(1 - (dx1**2))
                directions.append((dx1, dx2))
                directions.append((dx1, -1*dx2))
            
            temp_x1, temp_x2 = x1, x2
            for d1, d2 in directions:
                new_x1 = x1 + self.learning_rate * d1
                new_x2 = x2 + self.learning_rate * d2
                new_score = func([new_x1, new_x2])
                if new_score < best_score:
                    best_score = new_score
                    temp_x1 = new_x1
                    temp_x2 = new_x2
            x1, x2 = temp_x1, temp_x2
                
            output += f"{i:.5g}\t{x1:.5g}\t{x2:.5g}\t{best_score:.5g}\n"

        return output

    def coordinate_search(self, func) -> str:
        x1, x2 = self.x1, self.x2
        best_score = func([x1, x2])
        output = f"{0}\t{x1:.5g}\t{x2:.5g}\t{best_score:.5g}\n"
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        for i in range(1, self.iter + 1):
            improved = False

            temp_x1, temp_x2 = x1, x2
            for dx1, dx2 in directions:
                new_x1, new_x2 = x1 + (dx1 * self.learning_rate), x2 + (dx2 * self.learning_rate)
                new_score = func([new_x1, new_x2])

                if new_score < best_score:
                    best_score = new_score
                    temp_x1, temp_x2 = new_x1, new_x2
                    improved = True
            x1, x2 = temp_x1, temp_x2
            
            output += f"{i}\t{x1:.5g}\t{x2:.5g}\t{best_score:.5g}\n"
            if not improved:
                break
        
        return output

    def coordinate_descent(self, func) -> str:
        x1, x2 = self.x1, self.x2
        best_score = func([x1, x2])
        not_improved = 0
        output = f"{0}\t{x1:.5g}\t{x2:.5g}\t{best_score:.5g}\n"
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        for i in range(1, self.iter + 1):
            axis = directions[2:] if i % 2 == 0 else directions[:2]
            improved = False
            temp_x1, temp_x2 = x1, x2
            for dx1, dx2 in axis:
                new_x1, new_x2 = x1 + (dx1 * self.learning_rate), x2 + (dx2 * self.learning_rate)
                dir_score = func([new_x1, new_x2])
                if dir_score < best_score:
                    temp_x1, temp_x2 = new_x1, new_x2
                    best_score = dir_score
                    improved = True
                    not_improved = 0
            x1, x2 = temp_x1, temp_x2
            output += f"{i}\t{x1:.5g}\t{x2:.5g}\t{best_score:.5g}\n"
            if not improved:
                not_improved += 1
            if not_improved >= 2:
                break
        
        return output

def main():
    result = Minimizer(args.learning_rate, args.iteration_number, args.method_id, args.x1_val, args.x2_val, args.func_name)
    print(result.output)

    
if __name__ == "__main__":
    result = main()






 