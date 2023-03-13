inputs = [1, 2, 3]
weights = [0.2, 0.8, -0.5]
bias = 2

output = sum([input * weight for (input, weight) in zip(inputs, weights)]) + bias

print(output)
