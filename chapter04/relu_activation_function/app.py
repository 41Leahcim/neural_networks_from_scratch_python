from layers import Layer_Dense, Activation_ReLU
import time
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

start = time.time()

# Create dataset (samples = 100, classes = 3)
X, y = spiral_data(samples=2000000, classes=3)

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Create ReLU activation (to be used with Dense layer)
activation1 = Activation_ReLU()

# Perform a forward pass of our training data through this layer
dense1.forward(X)

# Forward pass through activation function.
# Takes in output from previous layer
activation1.forward(dense1.output)

# Let's see output of the first few samples
print(activation1.output[:5])
print(time.time() - start)
