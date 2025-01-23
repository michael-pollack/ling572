import argparse
import pandas as pd
import numpy as np
from math import log2
from sklearn.preprocessing import LabelEncoder
from collections import Counter

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--train', type=str, required=True, help='training data')
arg_parser.add_argument('--test', type=str, required=True, help='test data')
arg_parser.add_argument('--max_depth', required=True, type=str, help='max depth')
arg_parser.add_argument('--min_gain', required=True, type=str, help='min gain')
args = arg_parser.parse_args()

def parse_input(input) -> pd.DataFrame:
    data = []
    for line in input:
        parts = line.strip().split()
        doc_class = parts[0]
        words = {pair.split(":")[0]: pair.split(":")[1] for pair in parts[1:]}
        data.append({"class": doc_class, **words})
    return pd.DataFrame(data).fillna(0)

def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    le = LabelEncoder()
    df["class_encoded"] = le.fit_transform(df["class"])
    y_train = df.pop("class_encoded").values
    X_train = df.drop(columns=["class"])
    return X_train, y_train

def entropy(y: list[int]) -> float:
    counts = np.bincount(y)
    probs = counts / len(y)
    return -np.sum([p * log2(p) for p in probs if p > 0])  

def information_gain(y: list[int], y_left: list[int], y_right: list[int]) -> float:
    parent_entropy = entropy(y)      
    entropy_left = entropy(y_left)
    entropy_right = entropy(y_right)
    n = len(y)
    weighted_entropy = ((len(y_left) / n)  * entropy_left) + ((len(y_right) / n) * entropy_right)
    return parent_entropy - weighted_entropy

#Decision tree will split based on features
#We want total positive and negative instances of each feature for each class
#Use this to build decision tree
#Decision tree then is used on a set of features to try to determine what class it belongs to

class DecisionTree:
    def __init__(self, max_depth: int=None, min_gain: float=0.01) -> None:
        self.max_depth = max_depth
        self.min_gain = min_gain
        self.tree = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        self.tree = self.build_tree(X, y, depth=0)

    def build_tree(self, X: pd.DataFrame, y: np.ndarray, depth: int) -> dict:
        # Align y with X index only in the initial call
        if depth == 0:
            y = pd.Series(y, index=X.index)

        # Stopping condition: if all labels are the same or max depth is reached
        if depth == self.max_depth or len(set(y)) == 1:
            distribution = Counter(y)
            return {
                "leaf": True,
                "prediction": max(distribution, key=distribution.get),
                "distribution": distribution,
            }

        best_gain = 0
        best_feature = None
        best_split = None

        for feature in X.columns:
            left_indices = X[feature] == 0
            right_indices = X[feature] == 1

            # Slice X and y consistently
            X_left, X_right = X[left_indices], X[right_indices]
            y_left, y_right = y[left_indices], y[right_indices]

            gain = information_gain(y.values, y_left.values, y_right.values)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_split = (X_left, X_right, y_left.values, y_right.values)

        if best_gain < self.min_gain:
            distribution = Counter(y)
            return {
                "leaf": True,
                "prediction": max(distribution, key=distribution.get),
                "distribution": distribution,
            }

        left_tree = self.build_tree(*best_split[:2], depth + 1)
        right_tree = self.build_tree(*best_split[2:], depth + 1)
        return {
            "leaf": False,
            "feature": best_feature,
            "left": left_tree,
            "right": right_tree,
        }




def main():
    with open(args.train, 'r') as train_file:
        with open(args.test, 'r') as test_file:
            raw_train_data = parse_input(train_file)
            training_data, training_labels = preprocess_data(raw_train_data)
            raw_test_data = parse_input(test_file)
            testing_data, _ = preprocess_data(raw_test_data)
            tree = DecisionTree(max_depth=int(args.max_depth), min_gain=float(args.min_gain))
            tree.fit(training_data, training_labels)
            predictions = tree.predict(testing_data, return_full_node=True)
            for i, pred in enumerate(predictions):
                label_dist = " ".join(
                    f"{label} {pred['distribution'][label] / sum(pred['distribution'].values()):.2f}"
                    for label in sorted(pred["distribution"])
                )
                print(f"Line {i + 1}: Prediction {pred['prediction']} {label_dist}")


if __name__ == "__main__":
    result = main()