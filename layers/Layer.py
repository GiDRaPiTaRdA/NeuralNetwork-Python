from layers.ILayer import ILayer
from activationFunctions.sigmoid import SigmoidActivation
from neural_network import WEIGHT_MAX, WEIGHT_MIN


import numpy as np


class DenseLayer(ILayer):
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

    def save(self) -> dict[str, np.ndarray]:
        return {
            "weights": self.weights,
            "bias": self.bias
        }

    def load(self, data: dict[str, np.ndarray]) -> None:
        self.weights = data["weights"].copy()
        self.bias = data["bias"].copy()