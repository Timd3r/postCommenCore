import json

def predict_price():
    # Load the saved thetas
    try:
        with open("thetas.json", "r") as f:
            res = json.load(f)
            t0, t1 = res["t0"], res["t1"]
    except FileNotFoundError:
        # If training hasn't run, thetas must be 0
        t0, t1 = 0.0, 0.0
        #min_m, max_m = 0, 1

    # Prompt user for mileage
    try:
        val = input("Enter a mileage: ")
        mileage = float(val)
    except ValueError:
        print("Invalid input.")
        return

    # Scale the input mileage exactly like the training data
#    norm_mileage = (mileage - min_m) / (max_m - min_m) if max_m != min_m else 0

    # Apply the hypothesis formula
    # estimatePrice(mileage) = theta0 + (theta1 * mileage)
    print(t0)
    print(t1)
    print(mileage)
    estimate = t0 + (t1 * mileage)
    
    print(f"Estimated price: {estimate}")

if __name__ == "__main__":
    predict_price()