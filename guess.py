import ast
import numpy as np
from neural_network import NeuralNetwork, Layer

modelName = "models/model.npz"

network = NeuralNetwork([100,64,32,10])

network = NeuralNetwork([
    Layer(100, 64),
    Layer(64, 32),
    Layer(32, 10)
])

network.load(modelName)

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