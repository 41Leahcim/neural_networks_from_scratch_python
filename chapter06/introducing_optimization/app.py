from layers import Layer_Dense, Activation_ReLU, Activation_Softmax, Layer, Loss_CategoricalCrossentropy
import time
import nnfs
from nnfs.datasets import spiral_data
import numpy as np

nnfs.init()

start = time.time()

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

# Helper variables
lowest_loss = 1e10
best_weights = [layers[0].weights.copy(), layers[2].weights.copy()]
best_biases = [layers[0].biases.copy(), layers[2].biases.copy()]

iteration = 0
while True:
    # Generate new weights and biases
    layers[0].weights = 0.05 * np.random.randn(2, 3)
    layers[0].biases = 0.05 * np.random.randn(1, 3)
    layers[2].weights = 0.05 * np.random.randn(3, 3)
    layers[2].biases = 0.05 * np.random.randn(1, 3)

    # Forward the data through the model
    layers[0].forward(X)
    for i in range(1, len(layers)):
        layers[i].forward(layers[i - 1].output)
    
    # Perform a forward pass through the activation function.
    # It takes the output of the last layer (activation function) and return loss
    loss = loss_function.calculate(layers[-1].output, y)

    # Calculate accuracy from output of activation2 and targets.
    # Calculate values along first axis
    predictions = np.argmax(layers[-1].output, axis=1)
    accuracy = np.mean(predictions == y)

    # If loss if smaller - print and save weights
    if loss < lowest_loss:
        print(f"New set of weights found, iteration: {iteration}, loss: {loss}, acc: {accuracy}")
        best_weights = [layers[0].weights.copy(), layers[2].weights.copy()]
        best_biases = [layers[0].weights.copy(), layers[2].weights.copy()]
        lowest_loss = loss
    iteration += 1
