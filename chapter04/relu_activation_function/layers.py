import numpy as np

# Dense layer
class Layer_Dense:
    # Layer initialization
    def __init__(self, inputs: int, neurons: int):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(inputs, neurons)
        self.biases = np.zeros((1, neurons))

    # Forward pass
    def forward(self, inputs: list[list[float]]):
        # Calculate output values from inputs, weights, and biases
        self.output = np.dot(inputs, self.weights) + self.biases

# ReLU activation
class Activation_ReLU:
    # Forward pass
    def forward(self, inputs):
        # Calculate the output values from input
        self.output = np.maximum(0, inputs)
