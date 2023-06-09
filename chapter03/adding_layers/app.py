import numpy as np
import time

start = time.perf_counter()

inputs = [
    [1, 2, 3, 2.5],
    [2, 5, -1, 2],
    [-1.5, 2.7, 3.3, -0.8]
]

weights = [
    [0.2, 0.8, -0.5, 1],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]

biases = [2, 3, 0.5]

weights2 = [
    [0.1, -0.14, 0.5],
    [-0.5, 0.12, -0.33],
    [-0.44, 0.73, -0.13]
]

biases2 = [-1, 2, -0.5]

for i in range(100000):
    outputs1 = np.dot(inputs, np.array(weights).T) + biases
    outputs2 = np.dot(outputs1, np.array(weights2).T) + biases2

print(outputs2)

print(time.perf_counter() - start)
