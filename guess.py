import ast
import numpy as np
from layers.Layer import DenseLayer
from neural_network import NeuralNetwork

from layers.ConvLayer import ConvLayer
from layers.Layer import DenseLayer
from model_storage import ModelStorage

network = NeuralNetwork(
    ModelStorage("models/model_conv.npz"),
    [
        ConvLayer(8,3), # outputs 512 8 filters 3 by 3
        DenseLayer(512, 32),
        DenseLayer(32, 10)
    ])
network.load()

network.load()

def input_prediction(network):
    while True:
        text = input("Paste pixels (or 'exit'): ")

        if text.lower() == "exit":
            break

        try:
            pixels = ast.literal_eval(text)

            if not isinstance(pixels, list) or len(pixels) != 100:
                print("Expected 100 pixels")
                continue

            outputs = network.forwardActivations(pixels)[-1]

            for digit, score in enumerate(outputs):
                print(f"Digit {digit}: {score * 100:.2f}%")

            print(f"Prediction: {np.argmax(outputs)}\n")

        except (ValueError, SyntaxError, TypeError) as e:
            print(f"Invalid input: {e}")

input_prediction(network)