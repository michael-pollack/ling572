import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import Counter

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--train', type=str, required=True, help='training data')
arg_parser.add_argument('--test', type=str, required=True, help='test data')
arg_parser.add_argument('--model', type=str, required=True, help='model output file')
arg_parser.add_argument('--output', type=str, required=True, help='system output file')
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

class KNNClassifier:
    def __init__(self, train_data: str, k: int = 3, distance_metric: str='euclidean') -> None:
        self.k = k
        self.data, self.labels, self.label_map = parse_input(train_data)
        self.distance_metric = distance_metric
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
    
    def euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def cosine_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        dot_product = np.dot(x1, x2)
        norm_x1 = np.linalg.norm(x1)
        norm_x2 = np.linalg.norm(x2)
        return 1 - (dot_product / (norm_x1 * norm_x2)) if norm_x1 != 0 and norm_x2 != 0 else 1.0
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        predictions = [self.predict_single(x) for x in X_test]
        return np.array(predictions)
    
    def predict_single(self, x: np.ndarray) -> int:
        if self.distance_metric == 'euclidean':
            distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        elif self.distance_metric == 'cosine':
            distances = [self.cosine_distance(x, x_train) for x_train in self.X_train]
        else:
            raise ValueError(f"""
                             Hey you fool, we only take cosine or euclidean here.\n
                             You think you can just throw another distance metric into the mix?\n
                             Honestly, how dare you.
                             """)
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]