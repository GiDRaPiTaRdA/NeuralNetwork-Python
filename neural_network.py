import numpy as np
from abc import ABC, abstractmethod
from activationFunctions.sigmoid import SigmoidActivation

WEIGHT_MIN = -0.5
WEIGHT_MAX = 0.5

class NeuralNetwork:
    def __init__(self, pattern, activation_function=SigmoidActivation()):
        self.layers = []
        self.activation_function = activation_function

        for i in range(1, len(pattern)):
            layer = Layer(pattern[i - 1], pattern[i])
            self.layers.append(layer)

    def calculate_error(self, predicted, expected):
        return np.mean((predicted - expected) ** 2)

    def forward(self, inputs):
        values = np.array(inputs)

        for layer in self.layers:
            values = self.activation_function.function(
                layer.weights @ values + layer.bias
            )

        return values

    # Forward propagation: remember every layer's values
    def forwardActivations(self, inputs):
        activations = [np.array(inputs, dtype=float)]

        for layer in self.layers:
            values = self.activation_function.function(
                layer.weights @ activations[-1] + layer.bias
            )
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

            # Function derivative and error gradient
            delta = error * self.activation_function.derivative(output)

            # Propagate error backward BEFORE updating weights
            error = layer.weights.T @ delta

            # Update weights and biases
            layer.update(delta, previous, learning_rate)

        # Mean squared error at the output layer
        return self.calculate_error(activations[-1], expected)

class Layer:
    def __init__(self, input_count, neuron_count):
        self.weights = np.random.uniform(
            WEIGHT_MIN, WEIGHT_MAX, (neuron_count, input_count)
        )
        self.bias = np.zeros(neuron_count)

    def update(self, delta, previous, learning_rate):
        self.weights -= learning_rate * np.outer(delta, previous)
        self.bias -= learning_rate * delta

class TrainCase:
    def __init__(self, inputs, expected):
        self.inputs = np.array(inputs, dtype=float)
        self.expected = np.array(expected, dtype=float)