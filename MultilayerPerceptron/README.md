# 🧬 Multilayer Perceptron - Breast Cancer Detection

A from-scratch implementation of a Neural Network to classify breast cancer tumors as **Malignant** or **Benign** based on 30 cellular measurements.

---

## 📋 Features

- **Custom MLP Architecture:** Fully configurable hidden layers.
- **Numerical Stability:** Implements He Initialization and row-wise Softmax to prevent mathematical explosions.
- **Optimization:** Features Learning Rate Decay to ensure smooth convergence.
- **Monitoring:** Generates real-time Loss and Accuracy curves for both Training and Validation sets.

---

## 🚀 How to Run

### 1. Data Preparation

First, shuffle and split the raw dataset into training (80%) and testing (20%) sets:

```bash
python3 split_data.py
```

### 2. Training

Train the model. This will display the learning curves and save the optimized weights to `data/weights.json`:

```bash
python3 train.py
```

- The training uses the `data_train.csv` file and calculates validation metrics using `data_test.csv` to monitor for overfitting.

### 3. Prediction

Run the final evaluation on the test set using the saved weights:

```bash
python3 predict.py
```

---

## 📊 Results

- **Consistency:** The training and validation curves stay close, indicating strong generalization.
- **Accuracy:** Typically achieves **95% - 98%** accuracy on the unseen test set.