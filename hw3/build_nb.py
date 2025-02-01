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
arg_parser.add_argument('--model', type=str, required=True, help='model output file')
arg_parser.add_argument('--output', type=str, required=True, help='system output file')
arg_parser.add_argument('--cpdelta', type=str, required=True, help='conditional probability delta')
arg_parser.add_argument('--prdelta', type=str, required=True, help='class prior delta')
args = arg_parser.parse_args()

def parse_input(input) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    raw_data = []
    for line in input:
        parts = line.strip().split()
        doc_class = parts[0]
        words = {pair.split(":")[0]: 1 for pair in parts[1:]}
        raw_data.append({"this_class": doc_class, **words})
    data = pd.DataFrame(raw_data).fillna(0)
    return preprocess_data(data)

def preprocess_data(data: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    le = LabelEncoder()
    data["class_encoded"] = le.fit_transform(data["this_class"])
    labels_train = data.pop("class_encoded").values
    data_train = data.drop(columns=["this_class"])
    return data_train, labels_train, le.classes_

class NBClassifier:
    def __init__(self, train_data: str, model_file: str, prdelta: float=0, cpdelta: float=1) -> None:
         self.data, self.labels, self.label_map = parse_input(train_data)
         self.class_priors = {}
         self.feature_likelihoods = {}
         self.classes = None
         self.fit(self.data, self.labels, prdelta, cpdelta)
         self.print_model(model_file)

    def fit(self, data: pd.DataFrame, labels: np.ndarray, prdelta: float=0, cpdelta: float=1) -> None:
        #alphabetically sorted class list
        self.classes = np.sort(np.unique(labels))
        total_samples = data.shape[0]
        for cls in self.classes:
            class_samples = data[labels == cls]
            self.class_priors[cls] = (len(class_samples) + prdelta) / (total_samples + (len(self.classes) * prdelta))
            feature_counts = np.sum(class_samples, axis=0)
            total_feature_count = np.sum(feature_counts)
            self.feature_likelihoods[cls] = (feature_counts + cpdelta) / (total_feature_count + (cpdelta * data.shape[1]))
    
    def print_model(self, model_file: str) -> None:
        with open(model_file, 'w') as file:
            output = f"%%%%% prior prob P(c) %%%%%\n"
            for cls in self.class_priors:
                output += f"{self.label_map[cls]} {self.class_priors[cls]} {np.log10(self.class_priors[cls])}\n"
            output += f"%%%%% conditional prob P(f|c) %%%%%\n"
            for cls in self.feature_likelihoods:
                output += f"%%%%% conditional prob P(f|c) c={self.label_map[cls]} %%%%%\n"
                probs = self.feature_likelihoods[cls]
                for feature, prob in probs.items():
                    output += f"{feature}\t{self.label_map[cls]}\t{prob}\t{np.log10(prob)}\n"
            file.write(output)

    def predict(self, data: pd.DataFrame) -> tuple[np.ndarray, dict]:
        predictions = []
        full_predictions = {}
        count = 0
        for sample in data.values:
            class_probabilities = {}
            for cls in self.classes:
                log_prob = np.log10(self.class_priors[cls]) + np.sum(np.log10(self.feature_likelihoods[cls]) * sample)
                class_probabilities[cls] = log_prob
            predictions.append(max(class_probabilities, key=class_probabilities.get))
            full_predictions[count] = class_probabilities
            count += 1
        return np.array(predictions), full_predictions
    
    def run_test(self, test: str, output_file: str) -> None:
        train_preds, full_train_preds = self.predict(self.data)
        acc = self.run_acc(train_preds, self.labels, True)
        test_data, test_labels = self.process_test_data(test)
        test_preds, full_test_preds = self.predict(test_data)
        acc += self.run_acc(test_preds, test_labels)
        print(acc)
        self.print_predictions(test_preds, full_test_preds, output_file)
    
    def print_predictions(self, predictions: np.ndarray, full_predictions: dict, output_file: str) -> None:
        output = ""
        with open(output_file, 'w') as out_file:
            for pred_index in range(len(predictions)):
                prediction = self.label_map[predictions[pred_index]]
                full_pred = full_predictions[pred_index]
                output += f"array:{pred_index} {prediction}"
                sorted_classes = sorted(full_pred, key=full_pred.get, reverse=True)
                for cls in sorted_classes:
                    output += f" {self.label_map[cls]} {full_pred[cls]}"
                output += "\n"
            out_file.write(output)
    
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
        acc_output += f"{acc_label} accuracy: {total_acc}\n"

        return acc_output

    #account for differences in shape between train and test data
    def process_test_data(self, input: str) -> pd.DataFrame:
        raw_test_data, test_labels, _ = parse_input(input)
        return raw_test_data.reindex(columns=self.data.columns, fill_value=0), test_labels

    
def main() -> None:
    with open(args.train, 'r') as train:
        with open(args.test, 'r') as test:
            classifier = NBClassifier(train, args.model, float(args.prdelta), float(args.cpdelta))
            classifier.run_test(test, args.output)


if __name__ == "__main__":
    result = main()