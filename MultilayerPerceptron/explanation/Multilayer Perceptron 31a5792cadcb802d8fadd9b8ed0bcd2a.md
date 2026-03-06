# Multilayer Perceptron

## Part 1: The Anatomy of a Neuron

Think of a single neuron as a **decision-maker**. It takes several pieces of information, weighs them based on importance, and then decides how much "signal" to pass forward.

### 1. The Inputs (x) and Weights (w)

Each neuron receives multiple inputs (x0,x1,x2...). For your project, these will eventually be the features from the breast cancer dataset (like cell radius or texture).

Every input has a corresponding **weight** (w). The weight represents the **strength** or importance of the specific input. If a weight is high, that input has a large impact on the final decision; if it's near zero, the neuron mostly ignores it.

### 2. The Bias (b)

The **bias** is an extra input that is always equal to 1. It allows you to "shift" the activation function left or right. Essentially, the bias represents how easy or hard it is to get the neuron to "fire," regardless of the specific inputs.

### 3. Step 1: The Weighted Sum

The neuron first performs a bit of linear algebra. It multiplies every input by its weight, adds them all together, and then adds the bias.
The formula looks like this:

![image.png](Multilayer%20Perceptron/image.png)

### 4. Step 2: The Activation Function (f)

If we just used the weighted sum, our network would just be a series of simple linear equations. To learn complex patterns (like identifying cancer), we need **non-linearity**.

We pass the weighted sum through an **activation function**. This function acts as a threshold. It determines if the signal is strong enough to be passed to the next layer. Common examples you might use include:

- **Sigmoid:** Squashes the output between 0 and 1.
- **ReLU (Rectified Linear Unit):** Passes positive values through but turns all negative values to zero.

## Part 2: The Architecture & Forward Pass

In this project, you are building a **Multilayer Perceptron (MLP)**. It is a "feedforward" network, meaning the data flows in one direction: from the input layer, through the hidden layers, to the output layer.

### 1. The Input Layer

Your dataset has 32 columns, with one being the label (M or B) and the rest being features. This means your input layer will receive the features of a cell nucleus, such as radius or texture. In the diagram these are labeled x0,x1,x2,x3.

### 2. The Hidden Layers

The project requires your implementation to have **at least two hidden layers** by default.

- **Fully Connected:** These are "dense" layers, meaning every neuron in one layer is connected to every neuron in the next.
- **Complexity:** These layers allow the network to represent complex, non-linear relationships in the data.

### 3. The Output Layer & Softmax

Since you are predicting whether cancer is malignant or benign, you have two possible classes.

- **The Output:** Your output layer consists of neurons representing these classes.
- **Softmax:** You must implement the **softmax function** on the output layer.
    - **The Logic:** Softmax turns the raw scores from the last layer into a **probabilistic distribution**. Instead of just giving a random number, it might say there is a 0.90 (90%) probability of "Malignant" and a 0.10 (10%) probability of "Benign."

### 4. The "Forward Pass" (Feedforward)

This is the process of moving data through the network to get a prediction:

1. **Input:** Features from the Wisconsin dataset enter the first layer.
2. **Calculation:** Each neuron calculates its weighted sum + bias and applies its activation function.
3. **Propagation:** The output of one layer becomes the input for the next.
4. **Final Result:** The output layer provides the final probability using Softmax.

![image.png](Multilayer%20Perceptron/image%201.png)

## Part 3: The Learning Process (The "Brain")

If the **Forward Pass** is the network making a guess, **Part 3** is the teacher telling the network how wrong it was and how to fix it. This involves three interconnected concepts.

### 1. The Loss Function (Binary Cross-Entropy)

After the network makes its prediction (e.g., 85% Malignant), we compare it to the **actual** truth from the dataset.
The **Loss Function** calculates a single number that represents the "error" or "pain" of being wrong.

- If the network is 90% sure it's cancer and it *is* cancer, the Loss is very low.
- If the network is 90% sure it's cancer and it's actually *benign*, the Loss is very high.
Your goal is to make this Loss number as small as possible.

### 2. Back propagation (The "Blame" Phase)

Once we have that error (Loss), we move **backward** through the network—from the output layer back toward the input.

- We use **derivatives** (calculus) to figure out how much each specific weight and bias contributed to the error.
    - Think of it as "assigning blame." If a specific neuron's weight caused the network to be confidently wrong, that weight gets a "memo" saying it needs to change.

### 3. Gradient Descent (The "Update" Phase)

Now that we know *which* weights are to blame, we need to change them. **Gradient Descent** is the algorithm that updates the weights.

- It takes a small step in the direction that **decreases** the Loss.
- It uses a **Learning Rate** (a small number like 0.03) to make sure these steps aren't too big. If the steps are too big, the network might "overshoot" the solution and never learn anything.

## Part 4: Data & Evaluation (The Finish Line)

How to prove your model actually works. This involves two final steps.

### 1. The Train/Validation Split

You cannot test your model on the same data it learned from. That would be like a student seeing the exact questions and answers *before* an exam, they aren't "learning," they are just "memorizing."

- **Training Set:** The data the model uses to update its weights (The "Study Guide").
- **Validation Set:** A separate portion of data the model never sees during training. We use this to test if the model can predict cancer on **new, unseen patients** (The "Exam").

### 2. Learning Curves (Loss & Accuracy)

As you code, you will generate two graphs. These are the "heart monitors" of your project.

- **The Loss Curve:** This should go **down** over time. If it goes up, your learning rate is too high or your math is wrong.
- **The Accuracy Curve:** This should go **up** over time. It tells you the percentage of correct guesses (e.g., "Our model is 94% accurate").

### 3. The Goal

If your **Training Loss** is very low but your **Validation Loss** is very high ****(”low exam score”), your model is **Overfitting**. This means it memorized the training data but fails to understand general patterns. Your goal is to get both curves to settle at a low, stable point.