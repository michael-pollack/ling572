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
    raw_data = []
    for line in input:
        parts = line.strip().split()
        doc_class = parts[0]
        words = {pair.split(":")[0]: 1 for pair in parts[1:]}
        raw_data.append({"this_class": doc_class, **words})
    data = pd.DataFrame(raw_data).fillna(0)
    return data

def preprocess_data(data: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    le = LabelEncoder()
    data["class_encoded"] = le.fit_transform(data["this_class"])
    labels_train = data.pop("class_encoded").values
    data_train = data.drop(columns=["this_class"])
    return data_train, labels_train

def entropy(labels: np.ndarray) -> float:
    counts = np.bincount(labels)
    probs = counts / len(labels)
    return -np.sum([p * log2(p) for p in probs if p > 0])  

def information_gain(parent: np.ndarray, left_child: np.ndarray, right_child: np.ndarray) -> float:
    parent_entropy = entropy(parent)      
    entropy_left = entropy(left_child)
    entropy_right = entropy(right_child)
    n = len(parent)
    weighted_entropy = ((len(left_child) / n)  * entropy_left) + ((len(right_child) / n) * entropy_right)
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

    def fit(self, data: pd.DataFrame, labels: np.ndarray) -> None:
        self.tree = self.build_tree(data, labels, depth=0)

    def build_tree(self, data: pd.DataFrame, labels: np.ndarray, depth):
        if depth == self.max_depth or len(set(labels)) == 1:
            return {"leaf": True, "prediction": max(set(labels), key=list(labels).count)}

        best_gain = 0
        best_feature = None
        best_split = None

        for feature in data.columns:
            left_indices = data[feature] == 0
            right_indices = data[feature] == 1
            labels_left, labels_right = labels[left_indices], labels[right_indices]
            gain = information_gain(labels, labels_left, labels_right)

            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_split = (data[left_indices], data[right_indices], labels_left, labels_right)

        if best_gain < self.min_gain:
            return {"leaf": True, "prediction": max(set(labels), key=list(labels).count)}

        left_tree = self.build_tree(best_split[0], best_split[2], depth + 1)
        right_tree = self.build_tree(best_split[1], best_split[3], depth + 1)
        return {"leaf": False, "feature": best_feature, "left": left_tree, "right": right_tree}

    def predict(self, data: pd.DataFrame, return_full_node: bool = False) -> list:
        def traverse(node, row):
            if node["leaf"]:
                return node if return_full_node else node["prediction"]
            if row[node["feature"]] == 0:
                return traverse(node["left"], row)
            else:
                return traverse(node["right"], row)
        return [traverse(self.tree, row) for _, row in data.iterrows()]


def main():
    print("Starting")
    with open(args.train, 'r') as train_file:
        with open(args.test, 'r') as test_file:
            raw_train_data = parse_input(train_file)
            raw_test_data = parse_input(test_file)
            print(raw_train_data)
            training_data, training_labels = preprocess_data(raw_train_data)
            testing_data, _ = preprocess_data(raw_test_data)
            tree = DecisionTree(max_depth=int(args.max_depth), min_gain=float(args.min_gain))
            tree.fit(training_data, training_labels)
            # predictions = tree.predict(testing_data, return_full_node=True)
            # for i, pred in enumerate(predictions):
            #     label_dist = " ".join(
            #         f"{label} {pred['distribution'][label] / sum(pred['distribution'].values()):.2f}"
            #         for label in sorted(pred["distribution"])
            #     )
            #     print(f"Line {i + 1}: Prediction {pred['prediction']} {label_dist}")


if __name__ == "__main__":
    result = main()