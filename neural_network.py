import numpy as np
from activationFunctions.sigmoid import SigmoidActivation
from abc import ABC, abstractmethod
from numpy.typing import NDArray

WEIGHT_MIN = -0.5
WEIGHT_MAX = 0.5

class ILayer(ABC):

    @abstractmethod
    def activate(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        pass

    @abstractmethod
    def backward(
        self,
        error: NDArray[np.float64],
        output: NDArray[np.float64],
        previous: NDArray[np.float64],
        learning_rate: float
    ) -> NDArray[np.float64]:
        pass

    def save(self) -> dict[str, np.ndarray]:
        return {}

    def load(self, data: dict[str, np.ndarray]) -> None:
        pass

class NeuralNetwork:
    def __init__(self, layers: list[ILayer]):
        self.layers = layers

    # Mean Squared Error (MSE)
    def calculate_error(self, predicted, expected):
        return np.mean((predicted - expected) ** 2)

    def forward(self, inputs):
        return self.forwardActivations(inputs)[-1]
    
    # Forward propagation: remember every layer's values
    def forwardActivations(self, inputs):
        activations = [np.array(inputs, dtype=float)]

        for layer in self.layers:
            values = layer.activate(activations[-1])
            activations.append(values)

        return activations

    def save(self, path="model.npz"):
        data = {}

        for i, layer in enumerate(self.layers):
            data[f"weights_{i}"] = layer.weights
            data[f"bias_{i}"] = layer.bias

        np.savez(path, **data)

    def load(self, path="model.npz"):
        with np.load(path) as data:
            for i, layer in enumerate(self.layers):
                layer.weights = data[f"weights_{i}"].copy()
                layer.bias = data[f"bias_{i}"].copy()

    def train(self, inputs, expected, learning_rate=0.1):
        # Forward propagation: remember every layer's values
        activations = self.forwardActivations(inputs)
        expected = np.array(expected, dtype=float)

        # Error at the output layer (last item)
        error = activations[-1] - expected

        # Backpropagation (backward loop)
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            output = activations[i + 1]
            previous = activations[i]

            error = layer.backward(error, output, previous, learning_rate)

        # Mean squared error at the output layer
        return self.calculate_error(activations[-1], expected)

# 8x3x3
# 8 filters
# each filter is 3x3 matrix
class ConvLayer:
    def __init__(self, num_filters=8, filter_size=3):
        self.weights = np.random.randn(num_filters, filter_size, filter_size) * 0.1
        self.biases = np.zeros(num_filters)


class Layer(ILayer):
    def __init__(self, input_count, neuron_count, activationFunction=SigmoidActivation()):
        self.activation_fucntion = activationFunction
        self.weights = np.random.uniform(
            WEIGHT_MIN, WEIGHT_MAX, (neuron_count, input_count)
        )
        self.bias = np.zeros(neuron_count)

    def update(self, delta, previous, learning_rate):
        self.weights -= learning_rate * np.outer(delta, previous)
        self.bias -= learning_rate * delta

    def backward(self, error,output, previous, learning_rate):
         # Function derivative and error gradient
        delta = error * self.activation_fucntion.derivative(output)
        # Propagate error backward BEFORE updating weights
        error = self.weights.T @ delta
        # Update weights and biases
        self.update(delta, previous, learning_rate)
        return error

    def activate(self, inputs):
        output = self.weights @ inputs + self.bias
        activation = self.activation_fucntion.function(output)
        return activation
        

class TrainCase:
    def __init__(self, inputs, expected):
        self.inputs = np.array(inputs, dtype=float)
        self.expected = np.array(expected, dtype=float)