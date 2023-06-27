from layers import Layer_Dense, Activation_ReLU, Activation_Softmax, Layer, Loss_CategoricalCrossentropy
import time
import nnfs
from nnfs.datasets import spiral_data
import numpy as np

def calculate_accuracy(outputs, targets: np.ndarray):
    # Calculate prediction per sample
    predictions = np.argmax(outputs, axis=1)

    # Convert the targets into categorical labels, if targets were one-hot encoded
    if len(targets.shape) == 2:
        targets = np.argmax(targets, axis=1)
    
    # Calculate the percentage of predictions that were correct, True evaluates to 1; False to 0
    return np.mean(predictions == targets)

nnfs.init()

start = time.perf_counter()

# Create dataset (samples = 100, classes = 3)
X, y = spiral_data(samples=1_000_000, classes=3)

# Create a list of layer as the model
layers: list[Layer] = [
    Layer_Dense(2, 3),
    Activation_ReLU(),
    Layer_Dense(3, 3),
    Activation_Softmax()
]

# Define a loss function
loss_function = Loss_CategoricalCrossentropy()

# Forward the data through the model
layers[0].forward(X)
for i in range(1, len(layers)):
    layers[i].forward(layers[i - 1].output)

# Print the first 5 examples of the last layer
print(layers[-1].output[:5])


# Calculate and print the loss
print(loss_function.calculate(layers[-1].output, y))

# Calculate and print the accuracy
print(calculate_accuracy(layers[-1].output, y))

# Print the run-time
print(time.perf_counter() - start)
