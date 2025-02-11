import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import time

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
        self.label_set = set(self.labels)
        self.distance_metric = distance_metric
        self.data = np.array(self.pandas_data)
        self.norms = np.array([np.linalg.norm(sample) for sample in self.data])

    def run_acc(self, top_preds: np.ndarray, real_labels: np.ndarray, training: bool=False) -> str: 
        pred_table = [[0 for _ in self.label_map] for _ in self.label_map]
        correct = (real_labels == top_preds).sum()
        total = len(real_labels)
        total_acc = correct / total

        for index in range(len(top_preds)):
            real_label = real_labels[index]
            pred_label = top_preds[index]
            pred_table[real_label][pred_label] += 1
        #Construct the output file
        acc_output = f"""
Confusion matrix for the training data:
row is the truth, column is the system output

        \t\t
        """
        for label in self.label_map:
            acc_output += f"\t{label}"
        acc_output += "\n"

        for real_label in range(len(self.label_map)):
            acc_output += f"{self.label_map[real_label]}"
            for pred_label in range(len(self.label_map)):
                acc_output += f"\t{pred_table[real_label][pred_label]}"
            acc_output += f"\n"
        
        acc_label = "Training" if training else "Testing"
        acc_output += f"{acc_label} accuracy={total_acc}\n\n"

        return acc_output
    
    def euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def cosine_distance(self, x1: np.ndarray, x2: np.ndarray, norm_x2:float) -> float:
        dot_product = np.dot(x1, x2)
        norm_x1 = np.linalg.norm(x1)
        return 1 - (dot_product / (norm_x1 * norm_x2)) if norm_x1 != 0 and norm_x2 != 0 else 1.0

    def run_test(self, test_file: str, output_file: str) -> str:
        with open(test_file, 'r') as in_file:
            with open(output_file, 'w') as write_file:
                test_data, test_labels, _ = parse_input(in_file)
                test_data = test_data.reindex(columns=self.pandas_data.columns, fill_value=0)
                test_data = np.array(test_data)
                train_start = time.process_time()
                train_preds = self.predict(self.data)
                _, top_train_preds = self.process_predictions(train_preds, True)
                acc = self.run_acc(top_train_preds, self.labels, True)
                train_end = time.process_time()
                acc += f"Total Training CPU time: {(train_end - train_start) / 60} minutes\n"
                test_start = time.process_time()
                test_preds = self.predict(test_data)
                printed_predictions, top_test_preds = self.process_predictions(test_preds)
                acc += self.run_acc(top_test_preds, test_labels)
                test_end = time.process_time()
                acc += f"Total Testing CPU time: {(test_end - test_start) / 60} minutes\n"
                write_file.write(printed_predictions)
        return acc
    
    def process_predictions(self, predictions: np.ndarray, training: bool=False) -> tuple[str, np.ndarray]:
        printed_predictions = ""
        top_preds = []
        for index in range(len(predictions)):
            pred = predictions[index]
            sorted_classes = sorted(pred, key=pred.get, reverse=True)
            top_preds.append(sorted_classes[0])
            if not training:
                printed_predictions += f"array:{index} {self.label_map[sorted_classes[0]]}"
                zeros = self.label_set - set(sorted_classes)
                for cls in sorted_classes:
                    printed_predictions += f"\t{self.label_map[cls]}\t{pred[cls]}"
                for zero in zeros:
                    printed_predictions += f"\t{self.label_map[zero]}\t{0.0}"
                printed_predictions += "\n"
        top_preds = np.array(top_preds)
        return printed_predictions, top_preds
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        predictions = [self.predict_single(x) for x in X_test]
        return predictions
    
    def predict_single(self, test_data: np.ndarray) -> int:
        if self.distance_metric == '1':
            distances = [self.euclidean_distance(test_data, train_data) for train_data in self.data]
        elif self.distance_metric == '2':
            distances = [self.cosine_distance(test_data, self.data[index], self.norms[index]) for index in range(len(self.data))]
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
        acc = classifier.run_test(args.test, args.output)
        print(acc)

if __name__ == "__main__":
    result = main()
