from layers import Layer_Dense, Activation_ReLU, Activation_Softmax
import time
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

start = time.perf_counter()

# Create dataset (samples = 100, classes = 3)
X, y = spiral_data(samples=2_000_000, classes=3)

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Create ReLU activation (to be used with Dense layer)
activation1 = Activation_ReLU()

# Create second Dense layer with 3 input features
# (as we take output of previous layer here) and 3 output features
dense2 = Layer_Dense(3, 3)

# Create Softmax activation (to be used with Dense layer)
activation2 = Activation_Softmax()

# Perform a forward pass of our training data through this layer
dense1.forward(X)

# Forward pass through activation function.
# Takes in output from first dense layer here
activation1.forward(dense1.output)

# Make a forward pass through second dense layer
# it takes in output of activation function of first layer as inputs
dense2.forward(activation1.output)

# Forward pass through activation function.
# Takes in output from second dense layer here
activation2.forward(dense2.output)

# Let's see output of the first few samples
print(activation2.output[:5])
print(time.perf_counter() - start)
