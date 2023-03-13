inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]

# pure Python
output = list(map(lambda input: max(0, input), inputs))
print(output)

# Numpy
import numpy as np

output = np.maximum(0, inputs)
print(output)


