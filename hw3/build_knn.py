import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import Counter

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--train', type=str, required=True, help='training data')
arg_parser.add_argument('--test', type=str, required=True, help='test data')
arg_parser.add_argument('--k', type=str, required=True, help='k value')
arg_parser.add_argument('--function', type=str, required=False, help='similarity function')
arg_parser.add_argument('--output', type=str, required=True, help='output file')
args = arg_parser.parse_args()

#This parse_input uses real values! Not binary!
def parse_input(input) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    raw_data = []
    for line in input:
        parts = line.strip().split()
        doc_class = parts[0]
        words = {}
        for pair in parts[1:]:
            split_pair = pair.split(":")
            words[split_pair[0]] = np.float64(split_pair[1])
        raw_data.append({"this_class": doc_class, **words})
    data = pd.DataFrame(raw_data).fillna(0)
    return preprocess_data(data)

def preprocess_data(data: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    le = LabelEncoder()
    data["class_encoded"] = le.fit_transform(data["this_class"])
    labels_train = data.pop("class_encoded").values
    data_train = data.drop(columns=["this_class"])
    return data_train, labels_train, le.classes_

class KNNClassifier:
    def __init__(self, train_data: str, k: int = 3, distance_metric: str = '1') -> None:
        self.k = k
        self.pandas_data, self.labels, self.label_map = parse_input(train_data)
        self.distance_metric = distance_metric
        self.data = np.array(self.pandas_data)
    
    def euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def cosine_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        dot_product = np.dot(x1, x2)
        norm_x1 = np.linalg.norm(x1)
        norm_x2 = np.linalg.norm(x2)
        return 1 - (dot_product / (norm_x1 * norm_x2)) if norm_x1 != 0 and norm_x2 != 0 else 1.0

    def run_test(self, test_file: str, output_file: str) -> str:
        with open(test_file, 'r') as in_file:
            with open(output_file, 'w') as write_file:
                test_data, _, _ = parse_input(in_file)
                test_data = test_data.reindex(columns=self.pandas_data.columns, fill_value=0)
                test_data = np.array(test_data)
                predictions = self.predict(test_data)
                printed_predictions = self.print_predictions(predictions)
                print(output_file)
                print(printed_predictions)
                write_file.write(printed_predictions)
    
    def print_predictions(self, predictions: np.ndarray) -> str:
        output = ""
        for index in range(len(predictions)):
            pred = predictions[index]
            sorted_classes = sorted(pred, key=pred.get, reverse=True)
            output += f"array:{index} {self.label_map[sorted_classes[0]]}"
            zeros = set(self.labels) - set(sorted_classes)
            for cls in sorted_classes:
                output += f"\t{self.label_map[cls]}\t{pred[cls]}"
            for zero in zeros:
                output += f"\t{self.label_map[zero]}\t{0.0}"
            output += "\n"
        return output
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        predictions = [self.predict_single(x) for x in X_test]
        return predictions
    
    def predict_single(self, test_data: np.ndarray) -> int:
        if self.distance_metric == '1':
            distances = [self.euclidean_distance(test_data, train_data) for train_data in self.data]
        elif self.distance_metric == '2':
            distances = [self.cosine_distance(test_data, train_data) for train_data in self.data]
        else:
            raise ValueError(f"""
                             Hey you fool, we only take \'cosine\' or \'euclidean\' here.\n
                             You think you can just throw another distance metric into the mix?\n
                             Honestly, how dare you.
                             """)
        # Get the indices of the k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # Get the labels of the k nearest neighbors
        k_nearest_labels = [self.labels[i] for i in k_indices]
        
        # Compute label probabilities
        label_counts = Counter(k_nearest_labels)
        total = sum(label_counts.values())
        probabilities = {label: count / total for label, count in label_counts.items()}
        
        return probabilities

def main() -> None:
    with open(args.train, 'r') as train:
        classifier = KNNClassifier(train, int(args.k), args.function)
        classifier.run_test(args.test, args.output)

if __name__ == "__main__":
    result = main()
