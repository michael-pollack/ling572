import argparse
import numpy as np

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--test_data', type=str)
arg_parser.add_argument('--model_file', type=str)
arg_parser.add_argument('--sys_output', type=str)

args = arg_parser.parse_args()

class Decoder:
    def __init__(self, model_file):
        self.sv_coef = None
        self.support_vectors = None
        self.rho = None
        self.kernel_type = 'linear'
        self.gamma = 1.0  
        self.degree = 3  
        self.coef0 = 0.0  
        self.load_model(model_file)

    def run_predict(self, test_data, test_labels, sys_output):
        print(test_data)
        predictions = self.predict(test_data)
        self.print_predictions(predictions, test_labels, sys_output)

    def print_predictions(self, predictions, test_labels, sys_output):
        accuracy = []
        output = ""
        for i in predictions:
            pred, fx = predictions[i]
            if pred == test_labels[i]:
                accuracy.append(1)
            else:
                accuracy.append(0)
            output += f"{test_labels[i]} {pred} {fx}\n"
        total_acc = sum(accuracy) / len(accuracy)
        print(f"Accuracy: {total_acc}")
        with open(sys_output, 'w') as output_file:
            output_file.write(output)
    
    def load_model(self, model_file):
        with open(model_file, 'r') as f:
            lines = f.readlines()
        
        sv_start = False
        labels = []
        support_vectors = []
        
        max_feature_index = 0  # Track max feature index from model

        for line in lines:
            line = line.strip()
            if line.startswith('svm_type'):
                assert line.split()[1] == 'c_svc', "Only C-SVC is supported."
            elif line.startswith('kernel_type'):
                self.kernel_type = line.split()[1]
            elif line.startswith('gamma'):
                self.gamma = float(line.split()[1])
            elif line.startswith('coef0'):
                self.coef0 = float(line.split()[1])
            elif line.startswith('degree'):
                self.degree = int(line.split()[1])
            elif line.startswith('rho'):
                self.rho = float(line.split()[1])
            elif line.startswith('SV'):
                sv_start = True
                continue
            
            if sv_start:
                parts = line.split()
                labels.append(float(parts[0]))
                vector = [float(x.split(":")[1]) for x in parts[1:]]
                indices = [int(x.split(":")[0]) for x in parts[1:]]
                
                if indices:  # Check for empty support vectors
                    max_feature_index = max(max_feature_index, max(indices))
                
                support_vectors.append(vector)

        # Ensure all support vectors are the same length
        for i in range(len(support_vectors)):
            while len(support_vectors[i]) < max_feature_index + 1:
                support_vectors[i].append(0.0)  # Pad with zeros

        self.sv_coef = np.array(labels)
        self.support_vectors = np.array(support_vectors, dtype=float)
        self.num_features = max_feature_index + 1  # Store for test alignment


    def kernel_function(self, x, y):
        if self.kernel_type == 'linear':
            return np.dot(x, y)
        elif self.kernel_type == 'rbf':
            return np.exp(-self.gamma * np.linalg.norm(x - y) ** 2)
        elif self.kernel_type == 'polynomial':
            return (self.gamma * np.dot(x, y) + self.coef0) ** self.degree
        elif self.kernel_type == 'sigmoid':
            return np.tanh(self.gamma * np.dot(x, y) + self.coef0)
        else:
            raise ValueError("Unsupported kernel type")

    def predict(self, X):
        predictions = []
        for x in X:
            print(x)
            decision_value = sum(self.sv_coef[i] * self.kernel_function(self.support_vectors[i], x)
                                 for i in range(len(self.sv_coef))) - self.rho
            prediction = 0 if decision_value >= 0 else 1
            predictions.append((prediction, decision_value))
        return predictions
        
def load_libsvm_test_data(file_path: str) -> np.ndarray:
    with open(file_path, "r") as f:
        lines = f.readlines()
    
    max_index = 0
    instances = []
    labels = []
    
    for line in lines:
        parts = line.strip().split()
        features = {}
        labels.append(int(parts[0]))
        
        for item in parts[1:]:  # Ignore the first column (label)
            index, value = item.split(":")
            index = int(index)
            value = float(value)
            features[index] = value
            max_index = max(max_index, index)
        
        instances.append(features)

    # Convert sparse representation to dense NumPy array
    num_samples = len(instances)
    num_features = max_index + 1  # Feature indices start from 0
    dense_data = np.zeros((num_samples, num_features))

    for i, features in enumerate(instances):
        for index, value in features.items():
            dense_data[i, index] = value
    return dense_data, labels

def main():
    decoder = Decoder(args.model_file)
    test_data, test_labels = load_libsvm_test_data(args.test_data)
    decoder.run_predict(test_data, test_labels, args.sys_output)
    
if __name__ == "__main__":
    result = main()
