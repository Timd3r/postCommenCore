import json
import numpy as np

def run_full_analysis():
    # 1. Load the data to test against
    try:
        data = np.genfromtxt("data.csv", delimiter=",", skip_header=1)
    except Exception:
        print("Error: data.csv not found.")
        return

    # 2. Load the trained thetas
    try:
        with open("thetas.json", "r") as f:
            res = json.load(f) # This is where your error was!
            t0, t1 = res["t0"], res["t1"]
            #min_m, max_m = res["min"], res["max"]
    except FileNotFoundError:
        print("Error: Run training first to generate thetas.json")
        return

    total_pct_error = 0
    m = len(data)

    print(f"{'Actual Price':<12} | {'Predicted':<12} | {'Error %'}\t| {'Difference'}")
    print("-" * 40)
    diff_avg = 0
    for i in range(m):
        km = data[i, 0]
        actual_price = data[i, 1]

        # Normalize the km based on the training min/max
        #norm_km = (km - min_m) / (max_m - min_m)
        
        # Predict: Price = theta0 + (theta1 * mileage)
        prediction = t0 + (t1 * km)
        
        # Calculate the percentage difference
        error_pct = abs((prediction - actual_price) / actual_price) * 100
        total_pct_error += error_pct
        diff = prediction - actual_price
        diff_avg += abs(diff)
        print(f"{actual_price:<12.2f} | {prediction:<12.2f} | {error_pct:.2f}%\t| {abs(diff):.2f}")

    # Final Result
    average_error = total_pct_error / m
    average_diff = diff_avg / m
    print("-" * 40)
    print(f"OVERALL MODEL ERROR: {average_error:.2f}%")
    print(f"MODEL ACCURACY: {100 - average_error:.2f}%")
    print(f"AVERAGE DIFFERENCE: {average_diff:.2f}")

if __name__ == "__main__":
    run_full_analysis()