import csv
import json
import numpy as np

def load_weights():
    weights = {}
    bias = {}
    with open('data/weights.json', 'r') as f:
        data = json.load(f)
        weights = data['weights']
        bias = data['biases']
        min_max = data['min_max']
    return weights, bias, min_max

def load_input_data():
    with open('data/data_test.csv', 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data

def normalize_data(data, min_vals, max_vals):
    for col in range(data.shape[1]):
        data[:, col] = (data[:, col] - min_vals[col]) / (max_vals[col] - min_vals[col] + 1e-8) # Add small value to prevent division by zero
    return data


def predict(input_data, model_data):
    # 1. Strip ID and Labels, convert to float
    X = np.array(input_data)[:, 2:].astype(float)
    
    # 2. IMPORTANT: Normalize X here!
    min_vals = model_data['min_max']['min']
    max_vals = model_data['min_max']['max']
    X = normalize_data(X, min_vals, max_vals)
        
    weights = model_data['weights']
    biases = model_data['biases']

    current_input = X
    # 3. Replicate the Forward Pass exactly
    for i in range(len(weights)):
        # Convert list back to numpy for math
        W = np.array(weights[i])
        b = np.array(biases[i])
        
        # Math: (Input * Weight) + Bias
        z = np.dot(current_input, W) + b
        
        # Activation: Use Sigmoid for hidden, Softmax for last
        if i < len(weights) - 1:
            current_input = 1 / (1 + np.exp(-z)) # Sigmoid
        else:
            # Softmax for the final layer
            exp_z = np.exp(z - np.max(z))
            current_input = exp_z / exp_z.sum(axis=1, keepdims=True)
            
    # 4. Pick the winner (0 or 1)
    return np.argmax(current_input, axis=1)

if __name__ == "__main__":
    model_data = {
        'weights': [],
        'biases': []
    }
    model_data['weights'], model_data['biases'], model_data['min_max'] = load_weights()
    
    input_data = load_input_data()
    
    predictions = predict(input_data, model_data)
    true_labels = np.array(input_data)[:, 1].astype(int)
    test_accuracy = np.mean(predictions == true_labels) * 100
    
    print("Predictions:", predictions)
    print("Test Accuracy:", test_accuracy, "%")