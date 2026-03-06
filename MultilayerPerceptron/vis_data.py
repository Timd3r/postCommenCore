import csv
import pandas as pd

if __name__ == "__main__":
    # Load the training data
    with open('data/data.csv', 'r') as f:
        reader = csv.reader(f)
        data = list(reader)

    # Convert to DataFrame for better visualization
    df = pd.DataFrame(data[1:], columns=data[0])  # Skip header row

    # Display the first few rows of the DataFrame
    print(df.head())
    print(df.describe())
    #find nan
    print(df.isna().sum())
