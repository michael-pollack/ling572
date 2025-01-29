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
    def __init__(self, train_data: str, model_file: str) -> None:
         print("Classifier Initiated")
         self.data, self.labels, self.label_map = parse_input(train_data)
         self.class_priors = {}
         self.feature_likelihoods = {}
         self.classes = None
         print("Beginning Training")
         self.fit(self.data, self.labels)
         self.print_model(model_file)

    def fit(self, data: pd.DataFrame, labels: np.ndarray) -> None:
        #alphabetically sorted class list
        self.classes = np.sort(np.unique(labels))
        total_samples = data.shape[0]
        for cls in self.classes:
            class_samples = data[labels == cls]
            self.class_priors[cls] = len(class_samples) / total_samples
            feature_counts = np.sum(class_samples, axis=0)
            total_feature_count = np.sum(feature_counts)
            self.feature_likelihoods[cls] = (feature_counts + 1) / (total_feature_count + data.shape[1])
    
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
        print("Beginning Predictions")
        predictions = []
        full_predictions = {}
        count = 0
        for sample in data.values:
            class_probabilities = {}
            for cls in self.classes:
                print(len(self.feature_likelihoods[cls]))
                log_prob = np.log10(self.class_priors[cls]) + np.sum(np.log10(self.feature_likelihoods[cls]) * sample)
                class_probabilities[cls] = log_prob
            predictions.append(max(class_probabilities, key=class_probabilities.get))
            full_predictions[count] = class_probabilities
            count += 1
        return np.array(predictions), full_predictions
    
    def run_test(self, test: str) -> None:
        test_data = self.process_test_data(test)
        predictions, full_predictions = self.predict(test_data)
        self.print_predictions(predictions, full_predictions)
    
    def print_predictions(self, predictions: np.ndarray, full_predictions: dict) -> None:
        output = ""
        for pred in range(len(predictions)):
            prediction = predictions[pred]
            full_pred = full_predictions[pred]
            output += f"array:{pred} {prediction}"
            for cls in self.classes:
                output += f" {cls} {full_pred[cls]}"
            output += "\n"
        print(output)

    #account for differences in shape between train and test data
    def process_test_data(self, input: str) -> pd.DataFrame:
        raw_test_data, _, _ = parse_input(input)
        return raw_test_data.reindex(columns=self.data.columns, fill_value=0)



    
def main() -> None:
    with open(args.train, 'r') as train:
        with open(args.test, 'r') as test:
            classifier = NBClassifier(train, args.model)
            classifier.run_test(test)


if __name__ == "__main__":
    result = main()