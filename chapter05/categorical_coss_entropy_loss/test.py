import math
import numpy as np
vector = False

if vector:
    # An example output from the output layer of the neural network
    softmax_output = [0.7, 0.1, 0.2]

    # Ground truth
    target_output = [1, 0, 0]

    # pure Python
    # Calculate loss
    loss = -sum(
        map(
            lambda args:
                math.log(args[0]) * args[1],
            zip(softmax_output, target_output)
        )
    )

    print(loss)

    # Numpy
    # Calculate loss
    loss = -np.sum(np.log(softmax_output) * target_output)

    print(loss)
else:
    # An example output from the previous layer
    softmax_output = np.array([
        [0.7, 0.1, 0.2],
        [0.1, 0.5, 0.4],
        [0.02, 0.9, 0.08]
    ])

    # Ground truth (dog, cat, cat)
    class_targets = np.array([0, 1, 1])

    # Retrieve the accuracies in "pure" Python ("pure", because we are using numpy arrays)
    output = list(map(lambda args: args[1][args[0]], zip(class_targets, softmax_output)))
    print(output)

    # Retrieve the accuracies with numpy
    print(softmax_output[range(len(softmax_output)), class_targets])

    # Calculate losses in "pure" Python ("pure", because we are using numpy arrays)
    output = sum(
        map(
            lambda args: -math.log(args[1][args[0]]),
            zip(class_targets, softmax_output)
        )
    ) / len(class_targets)
    print(output)

    # Calculate losses with numpy
    output = np.mean(
        -np.log(
            softmax_output[
                range(len(softmax_output)),
                class_targets
            ]
        )
    )
    print(output)

    class_targets = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0]
    ])

    # Target shape dependent calculation of loss in "pure" Python ("pure", because we are using numpy arrays)
    if len(class_targets.shape) == 1:
        average_loss = -sum(
            map(
                lambda args: math.log(args[1][args[0]]),
                zip(class_targets, softmax_output)
            )
        ) / len(class_targets)
    elif len(class_targets.shape) == 2:
        total_loss = 0
        for target_row, output_row in zip(class_targets, softmax_output):
            total_loss += math.log(sum(map(
                    lambda values:  values[0] * values[1],
                    zip(target_row, output_row)
                ))
            )
        average_loss = -total_loss / len(class_targets)
    print(average_loss)

    # Target shape dependent calculation of loss with numpy
    if len(class_targets.shape) == 1:
        correct_confidences = softmax_output[range(len(softmax_output)), class_targets]
    elif len(class_targets.shape) == 2:
        correct_confidences = np.sum(softmax_output * class_targets, axis=1)

    # Calculate average loss
    average_loss = np.mean(-np.log(correct_confidences))
    print(average_loss)
