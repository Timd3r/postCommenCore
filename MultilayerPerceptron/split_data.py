import csv
from sklearn.utils import shuffle

def load_data():
    with open('data/data.csv', 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    for i in range(len(data)):
        data[i][1] = 0 if data[i][1] == 'M' else 1
    return data

def randomize_data(data):
    data = shuffle(data)
    return data

def split_data(data, train_ratio):
    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

if __name__ == "__main__":
    data = load_data()
    data = randomize_data(data)
    train_ratio = 0.8
    train_data, test_data = split_data(data, train_ratio)
    
    # Save train data
    with open('data/data_train.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(train_data)
    
    # Save test data
    with open('data/data_test.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(test_data)