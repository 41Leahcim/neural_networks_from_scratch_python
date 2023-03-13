import numpy as np

# Create a base layer class for every layer to extend
class Layer:
    def forward(self, inputs):
        self.output = []
        pass

# Dense layer
class Layer_Dense(Layer):
    # Layer initialization
    def __init__(self, inputs: int, neurons: int):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(inputs, neurons)
        self.biases = np.zeros((1, neurons))

    # Forward pass
    def forward(self, inputs):
        # Calculate output values from inputs, weights, and biases
        self.output = np.dot(inputs, self.weights) + self.biases

# ReLU activation
class Activation_ReLU(Layer):
    # Forward pass
    def forward(self, inputs):
        # Calculate the output values from input
        self.output = np.maximum(0, inputs)

# Softmax activation
class Activation_Softmax(Layer):
    # Forward pass
    def forward(self, inputs):
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalize them for each sample
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

class Activation_Softmax(Layer):
    # Forward pass
    def forward(self, inputs):
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalize them for each sample
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

# Common loss class
class Loss(Layer):
    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y):
        # Calculate mean loss
        return np.mean(self.forward(output, y))

# Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):
    # Forward pass
    def forward(self, y_pred, y_true: np.ndarray):
        # Number of samples in a batch
        samples = len(y_pred)

        # Clip data to prevent division by 0
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values - only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        
        # Calculate losses
        return -np.log(correct_confidences)
