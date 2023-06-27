from layers import Layer_Dense
import time
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

start = time.perf_counter()

# Create dataset
X, y = spiral_data(samples=10000, classes=3)

for i in range(4000):
    # Create Dense layer with 2 input features and 3 output values
    dense1 = Layer_Dense(2, 3)

    # Perform a forward pass of our training data through this layer
    dense1.forward(X)

# Let's see output of the first few samples
print(dense1.output[:5])
print(time.perf_counter() - start)
