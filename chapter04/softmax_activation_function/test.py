# Values from the previous output when we described what a neural network is
layer_outputs = [4.8, 1.21, 2.385]

# Determinces whether numpy or pure Python should be used
use_numpy = True

if use_numpy: # Using numpy
    import numpy as np

    # For each value in a vector, calculate the exponential value
    exp_values = np.exp(layer_outputs)
    print(f"Exponential values:\n{exp_values}")

    # Normalize values
    norm_base = np.sum(exp_values)
    norm_values = exp_values / norm_base
    print(f"Normalized exponential values:\n{norm_values}")

    print(f"Sum of normalized values: {np.sum(norm_values)}")
else: # Using pure Python
    import math

    # For each value in a vector, calculate the exponential value
    exp_values = list(map(math.exp, layer_outputs))
    print(f"Exponential values:\n{exp_values}")

    # Normalize values
    norm_base = sum(exp_values)
    norm_values = list(map(lambda value: value / norm_base, exp_values))
    print(f"Normalized exponential values:\n{norm_values}")

    print(f"Sum of normalized values: {sum(norm_values)}")

