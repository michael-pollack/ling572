import argparse
import pandas as pd
import numpy as np
from math import log2
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import time

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--train', type=str, required=True, help='training data')
arg_parser.add_argument('--test', type=str, required=True, help='test data')
arg_parser.add_argument('--max_depth', required=True, type=str, help='max depth')
arg_parser.add_argument('--min_gain', required=True, type=str, help='min gain')
arg_parser.add_argument('--model', required=True, type=str, help='model output file')
arg_parser.add_argument('--output', required=True, type=str, help='system output file')
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
    return data_train, labels_train, le.classes_

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

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float: 
    pred_list = [pred['prediction'] for pred in y_pred]
    correct = (y_true == pred_list).sum()
    total = len(y_true)
    return correct / total

class DecisionTree:
    def __init__(self, max_depth: int=None, min_gain: float=0.01) -> None:
        self.max_depth = max_depth
        self.min_gain = min_gain
        self.tree = None
        self.label_map = None
        self.pred_to_real = {}

    def fit(self, data: pd.DataFrame, labels: np.ndarray, label_map: np.ndarray) -> None:
        self.tree = self.build_tree(data, labels, depth=0)
        self.label_map = label_map

    def build_tree(self, data: pd.DataFrame, labels: np.ndarray, depth):
        if depth == self.max_depth or len(set(labels)) == 1:
            distribution = Counter(labels)
            total_samples = sum(distribution.values())
            probabilities = {
                label: count / total_samples for label, count in distribution.items()
            }
            return {
                "leaf": True,  # Mark as a leaf node
                "prediction": max(distribution, key=distribution.get),  # Majority label
                "distribution": probabilities,  # Probabilities of labels
                "samples": total_samples,  # Total number of samples
            }

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

        if best_gain < self.min_gain or best_split is None:
            distribution = Counter(labels)
            total_samples = sum(distribution.values())
            probabilities = {
                label: count / total_samples for label, count in distribution.items()
            }
            return {
                "leaf": True,
                "prediction": max(distribution, key=distribution.get),
                "distribution": probabilities,
                "samples": total_samples,
            }

        left_tree = self.build_tree(best_split[0], best_split[2], depth + 1)
        right_tree = self.build_tree(best_split[1], best_split[3], depth + 1)
        return {
            "leaf": False,  # Not a leaf
            "feature": best_feature,  # Feature used for splitting
            "left": left_tree,  # Left subtree
            "right": right_tree,  # Right subtree
        }
    
    def print_tree(self, file: str, node=None, path="") -> None:
        if node is None:
            node = self.tree  # Start from the root of the tree

        if node["leaf"]:
            # Extract label distribution and total number of samples
            label_distribution = node["distribution"]
            total_samples = node["samples"]

            # Format the label probabilities in alphabetical order
            label_str = " ".join(
                f"{self.label_map[label]} {label_distribution.get(label, 0):.1f}"
                for label in sorted(label_distribution)
            )

            # Print the path, total samples, and label distribution
            with open(file, 'a') as output:
                output.write(f"{path.strip('&')} {total_samples} {label_str}\n")
            return

        # Traverse the left subtree (feature == 0)
        self.print_tree(file, node["left"], path + f"!{node['feature']}&")

        # Traverse the right subtree (feature == 1)
        self.print_tree(file, node["right"], path + f"{node['feature']}&")


    def predict(self, data: pd.DataFrame, return_full_node: bool = False) -> list:
        def traverse(node, row):
            if node["leaf"]:
                return node if return_full_node else node["prediction"]
            if node["feature"] not in row or row[node["feature"]] == 0:
                return traverse(node["left"], row)
            else:
                return traverse(node["right"], row)
        return [traverse(self.tree, row) for _, row in data.iterrows()]
    
    def print_predictions(self, file: str, predictions: list, real_labels: list, real_label_map: list) -> None:
        prediction_table = [[0 for _ in range(len(self.label_map))] for _ in range(len(real_label_map))]
        with open(file, 'w') as output:
            for i, pred in enumerate(predictions):
                label_dist = " ".join(
                    f"{self.label_map[label]} {pred['distribution'][label] / sum(pred['distribution'].values()):.4f}"
                    for label in sorted(pred["distribution"])
                )
                this_pred = pred['prediction']
                real_label = real_labels[i]
                prediction_table[real_label][this_pred] += 1
                output.write(f"array:{i}  {label_dist}\n")
        return prediction_table
    
    def print_pred_table(self, pred_table: list[list], real_label_map: list) -> None:
        table = "\t\t\t"
        for label in self.label_map:
            table += f"{label} "
        table += "\n"
        for i in range(len(pred_table)):
            row = f"{real_label_map[i]}\t"
            for pred_label in pred_table[i]:
                row += f"{pred_label} "
            row += f"\n"
            table += row
        print(table)
        
        


def main():
    start = time.process_time()
    with open(args.train, 'r') as train_file:
        with open(args.test, 'r') as test_file:
            raw_train_data = parse_input(train_file)
            raw_test_data = parse_input(test_file)
            training_data, training_labels, label_map = preprocess_data(raw_train_data)
            testing_data, testing_labels, testing_label_map = preprocess_data(raw_test_data)
            tree = DecisionTree(max_depth=int(args.max_depth), min_gain=float(args.min_gain))
            tree.fit(training_data, training_labels, label_map)
            #wipe the model output file
            with open(args.model, 'w') as wipe:
                pass
            tree.print_tree(args.model)
            train_predictions = tree.predict(training_data, return_full_node=True)
            test_predictions = tree.predict(testing_data, return_full_node=True)
            train_accuracy = accuracy_score(training_labels, train_predictions)
            test_accuracy = accuracy_score(testing_labels, test_predictions)
            train_pred_table = tree.print_predictions("trash.txt", train_predictions, training_labels, label_map)
            pred_table = tree.print_predictions(args.output, test_predictions, testing_labels, testing_label_map)

            print(f"Confusion matrix for the training data:\n row is the truth, column is the system output\n\n")
            tree.print_pred_table(train_pred_table, label_map)
            print(f"Training Accuracy={train_accuracy}")

            print(f"Confusion matrix for the testing data:\n row is the truth, column is the system output\n\n")
            tree.print_pred_table(pred_table, testing_label_map)
            print(f"Testing Accuracy={test_accuracy}")

    end = time.process_time()
    print(f"Total CPU time: {(end - start) / 60} minutes")



if __name__ == "__main__":
    result = main()