from activationFunctions.relu_function import ReLUActivation
from layers.ILayer import ILayer

import numpy as np


class ConvLayer(ILayer):
    def __init__(self, num_filters=8, filter_size=3, activationFunction=ReLUActivation()):
        self.activation_fucntion = activationFunction
        self.filters_number = num_filters
        self.filters_size = filter_size
        self.filters = np.random.randn(
            num_filters, filter_size, filter_size) * 0.1
        self.biases = np.zeros(num_filters)

    def save(self) -> dict[str, np.ndarray]:
        return {
            "filters": self.filters,
            "biases": self.biases
        }

    def load(self, data: dict[str, np.ndarray]) -> None:
        self.filters = data["filters"].copy()
        self.biases = data["biases"].copy()

    def backward(self, error, output, previous, learning_rate):
        # 1. Undo flattening
        error = error.reshape(
            self.filters_number,
            previous.shape[0] - self.filters_size + 1,
            previous.shape[1] - self.filters_size + 1
        )

        # 2. Apply ReLU derivative
        activated_output = output.reshape(error.shape)
        delta = error * (activated_output > 0)

        # 3. Extract input windows
        windows = np.lib.stride_tricks.sliding_window_view(
            previous,
            (self.filters_size, self.filters_size)
        )

         # 4. Calculate filter gradients
        filters_gradient = np.einsum(
            "fij,ijxy->fxy",
            delta,
            windows
        )

        # 5. Calculate bias gradients
        biases_gradient = np.sum(delta, axis=(1, 2))

        # 6. Calculate gradient for the previous layer
        previous_error = np.zeros_like(previous)

        for f in range(self.filters_number):
            for i in range(delta.shape[1]):
                for j in range(delta.shape[2]):
                    previous_error[
                        i:i + self.filters_size,
                        j:j + self.filters_size
                    ] += delta[f, i, j] * self.filters[f]

        # 7. Update parameters
        self.filters -= learning_rate * filters_gradient
        self.biases -= learning_rate * biases_gradient

        return previous_error

    def activate(self, pixels):
        windows = np.lib.stride_tricks.sliding_window_view(
            pixels, 
            (self.filters_size, self.filters_size))
        output = np.einsum("ijxy,fxy->fij", windows, self.filters)
        output += self.biases[:, None, None]

        # ReLu
        output = self.activation_fucntion.function(output)

         # 8x8x8 flattened into 512
        flattened = output.flatten()

        return flattened

    def update(self, delta, previous, learning_rate):
        self.weights -= learning_rate * np.outer(delta, previous)
        self.bias -= learning_rate * delta
