import csv
import json
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

class MultilayerPerceptron:
    def __init__(self, layer_sizes):
        # layer_sizes could be [30, 24, 24, 2]
        # You will initialize Weights and Biases here randomly
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            limit = np.sqrt(2/layer_sizes[i])
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * limit)
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True)) 
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def forward_pass(self, X):
        # The math: activation( (X * W) + b )
        self.activations = [X]
        
        # saves the z values for backpropagation
        self.zs = []

        # Loop through each layer and calculate the activations
        current_input = X
        for i in range(len(self.weights)):
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            self.zs.append(z)
            if i == len(self.weights) - 1:
                current_input = self.softmax(z)
            else:
                current_input = self.sigmoid(z)
            self.activations.append(current_input)
        return current_input

    def backpropagation(self, X, y, learning_rate):
        # 1. INITIAL ERROR (Output Layer)
        # Assumes y is One-Hot encoded and output is Softmax
        # Logic: Guess - Truth
        error = self.forward_pass(X) - y 
        
        # 2. WALK BACKWARDS through the weight layers
        for i in reversed(range(len(self.weights))):
            # A. CALCULATE GRADIENTS
            # Logic: Input from the layer before (i) dot product with current error
            eW = np.dot(self.activations[i].T, error)
            eB = np.sum(error, axis=0, keepdims=True)
            
            # B. PREPARE ERROR FOR THE PREVIOUS LAYER
            # We do this BEFORE updating weights[i] to keep the math pure.
            if i > 0:
                # Logic: We use the weights BEFORE they are updated
                # and the Sigmoid derivative of the layer we are moving into.
                derivative = self.activations[i] * (1 - self.activations[i])
                next_layer_error = np.dot(error, self.weights[i].T) * derivative
            
            # C. UPDATE WEIGHTS AND BIASES
            # Now that we've used the old weights to pass the error back,
            # we can safely update them.
            self.weights[i] -= learning_rate * eW
            self.biases[i] -= learning_rate * eB
            
            # D. UPDATE ERROR POINTER
            # Set error to the one we just calculated for the next loop iteration
            if i > 0:
                error = next_layer_error
            


def load_data():
    with open('data/data_train.csv', 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data

def save_weights(mlp):
    weight_list = [w.tolist() for w in mlp.weights]
    bias_list = [b.tolist() for b in mlp.biases]
    min_max_values = {'min': mlp.min_vals.tolist(), 'max': mlp.max_vals.tolist()}
    with open('data/weights.json', 'w', newline='') as f:
        json.dump({'weights': weight_list, 'biases': bias_list, 'min_max': min_max_values}, f)

def normalize_data(data, min_vals, max_vals):
    for col in range(data.shape[1]):
        data[:, col] = (data[:, col] - min_vals[col]) / (max_vals[col] - min_vals[col] + 1e-8) # Add small value to prevent division by zero
    return data

def load_test_data(min_vals, max_vals):
    with open('data/data_test.csv', 'r') as f:
        reader = csv.reader(f)
        data = np.array(list(reader))
    
    X_test = data[:, 2:].astype(float)
    y_test = data[:, 1].astype(int)
    
    # Use the SAME min/max from the training set
    X_test = (X_test - min_vals) / (max_vals - min_vals + 1e-8)
    y_test_one_hot = np.eye(2)[y_test]
    
    return X_test, y_test_one_hot, y_test

def train_model(data):
    # 1. Convert the list of strings into a NumPy array
    data = np.array(data)
    
    # 2. Slice the data
    X = data[:, 2:].astype(float) 
    y = data[:, 1].astype(int)

    # 5. Initialize the Brain
    mlp = MultilayerPerceptron([30, 24, 24, 2])

    # Capture the specific Min/Max of THIS training set
    mlp.min_vals = np.min(X, axis=0)
    mlp.max_vals = np.max(X, axis=0)

    # 3. Normalize X (Crucial so weights don't explode!)
    X = normalize_data(X, mlp.min_vals, mlp.max_vals)
    
    # 4. Create the One-Hot Answer Key
    y_one_hot = np.eye(2)[y] 
    
    # 6. The Training Loop
    epochs = 1000
    learning_rate = 0.001

    loss_history = []    
    # 6. The Training Loop
    epochs = 1000
    initial_learning_rate = 0.005
    X_val, y_val_one_hot, y_val = load_test_data(mlp.min_vals, mlp.max_vals)

    # Initialize histories
    hist = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    
    for i in range(epochs):
        # Update weights
        mlp.backpropagation(X, y_one_hot, learning_rate)

        # 1. Training Metrics
        train_pred = mlp.forward_pass(X)
        hist['train_loss'].append(np.mean(np.square(train_pred - y_one_hot)))
        hist['train_acc'].append(np.mean(np.argmax(train_pred, axis=1) == y) * 100)

        # 2. Validation Metrics (No learning happens here)
        val_pred = mlp.forward_pass(X_val)
        hist['val_loss'].append(np.mean(np.square(val_pred - y_val_one_hot)))
        hist['val_acc'].append(np.mean(np.argmax(val_pred, axis=1) == y_val) * 100)

        if i % 100 == 0:
            print(f"Epoch {i} | Loss: {hist['train_loss'][-1]:.4f} | Acc: {hist['train_acc'][-1]:.2f}%")
            
    save_weights(mlp)
    return hist


if __name__ == "__main__":
    raw_data = load_data()
    h = train_model(raw_data)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Loss
    ax1.plot(h['train_loss'], label='Train Loss')
    ax1.plot(h['val_loss'], label='Val Loss', linestyle='--')
    ax1.set_title('Loss History')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('MSE')
    ax1.legend()

    # Plot 2: Accuracy
    ax2.plot(h['train_acc'], label='Train Acc')
    ax2.plot(h['val_acc'], label='Val Acc', linestyle='--')
    ax2.set_title('Accuracy History')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Percentage (%)')
    ax2.legend()

    plt.tight_layout()
    plt.show()