import numpy as np
from layers.ILayer import ILayer
from model_storage import ModelStorage

WEIGHT_MIN = -0.5
WEIGHT_MAX = 0.5

class NeuralNetwork:

    def __init__(self, storage: ModelStorage, layers: list[ILayer]):
        self.layers = layers
        self.storage = storage

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

    def save(self):
        self.storage.save(self.layers)

    def load(self):
        self.storage.load(self.layers)

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
class TrainCase:
    def __init__(self, inputs, expected):
        self.inputs = np.array(inputs, dtype=float)
        self.expected = np.array(expected, dtype=float)