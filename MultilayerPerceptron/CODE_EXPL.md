Code Logic Explanation

This document breaks down the mathematical and structural logic used in the implementation of this Multilayer Perceptron.
1. Network Architecture

The model uses a feed-forward architecture defined by layer_sizes (e.g., [30, 24, 24, 2]):

    Input Layer (30): Corresponds to the 30 features of the breast cancer dataset.

    Hidden Layers (24, 24): Two layers that extract complex patterns from the data using the Sigmoid activation function.

    Output Layer (2): Two neurons representing the probability of the tumor being Malignant or Benign, using the Softmax function.

2. Mathematical Stability & Initialization

To prevent the "Spike" issues and numerical explosions, the code implements two key strategies:

    He Initialization: Weights are initialized using np.random.randn and scaled by sqrt(2 / inputs). This ensures that the variance of the signals remains consistent across layers.

    Row-wise Softmax: When calculating probabilities, we subtract the maximum value of each row (np.max(z, axis=1)) before exponentiation to prevent "overflow" errors.

3. Forward Propagation

The forward pass follows the linear transformation formula:
Z=(X⋅W)+b

The result Z is then passed through an activation function (a):

    Sigmoid: σ(z)=1+e−z1​ (Used for hidden layers).

    Softmax: ∑ezj​ezi​​ (Used for the output layer to ensure probabilities sum to 100%).

4. Backpropagation & Optimization

The "learning" happens by calculating how much each weight contributed to the error and adjusting it:

    Error Calculation: We calculate the difference between the prediction and the true label.

    Gradient Descent: We calculate the derivative of the error with respect to each weight.

    Weight Update: We subtract a small portion of this gradient from the weights: W=W−(η⋅∇W), where η is the learning rate.

    Learning Rate Decay: The learning rate is multiplied by 0.9 every 100 epochs to allow the model to "settle" into the optimal solution without overshooting.

5. Data Normalization

The 30 features in the dataset have very different scales (e.g., area vs. smoothness). We use Min-Max Scaling to squash all values between 0 and 1:
Xnorm​=max−minX−min​

This ensures that features with larger numbers don't unfairly dominate the weight updates.