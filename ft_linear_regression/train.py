import numpy as np
import json

def train_model():
    # Load dataset
    try:
        data = np.genfromtxt("data.csv", delimiter=",", skip_header=1)
    except Exception:
        print("Error: Could not read data.csv")
        return

    mileage = data[:, 0]
    price = data[:, 1]
    m = len(data) # m is the number of data points

    # Feature Scaling (Normalization) 
    # This prevents 'nan' by keeping numbers between 0 and 1
    min_m, max_m = np.min(mileage), np.max(mileage)
    norm_m = (mileage - min_m) / (max_m - min_m)
    print("Data loaded and normalized.\n norm_mileage: ", norm_m)

    # Initializing variables as required [cite: 81]
    theta0 = 0.0
    theta1 = 0.0
    learning_rate = 0.1 # Adjust this to change how fast it learns

    for _ in range(10000): # Number of iterations
        sum0 = 0
        sum1 = 0
        
        # Calculate the gradient for each data point [cite: 85, 86]
        for i in range(m):
            # estimatePrice = theta0 + (theta1 * mileage) [cite: 80, 88]
            prediction = theta0 + (theta1 * norm_m[i])
            error = prediction - price[i]
            
            sum0 += error
            sum1 += error * norm_m[i]

        # Simultaneous update [cite: 89]
        # formula: theta = theta - (learningRate * 1/m * sum) [cite: 86]
        theta0 -= (learning_rate * (1/m) * sum0)
        theta1 -= (learning_rate * (1/m) * sum1)

    # Save variables for the prediction program [cite: 84]
    with open("thetas.json", "w") as f:
        json.dump({"t0": theta0, "t1": theta1, "min": min_m, "max": max_m}, f)
    print("Training complete. Thetas saved.")

if __name__ == "__main__":
    train_model()