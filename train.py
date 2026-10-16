"""Simple matrix multiplication example. No neural network implementation."""
import numpy as np
import ast
from layers.Layer import DenseLayer
from neural_network import NeuralNetwork
from data.data_loader import DataLoader
from model_storage import ModelStorage
from layers.ConvLayer import ConvLayer

def train_network(network, cases, epochs, learning_rate):
    for epoch in range(epochs):
        total_error = 0

        for case in cases:
            error = network.train(
                np.asarray(case.inputs).reshape(10, 10),
                case.expected,
                learning_rate
            )

            total_error += error

        if epoch % (epochs // 10) == 0:
            print(f"Epoch {epoch}, Error: {total_error / len(cases):.6f}")

    print("Training completed.")

def visualize_prediction(inputs, network):
    prediction = network.forwardActivations(inputs)[-1]
    percentages = prediction / np.sum(prediction) * 100

    for digit, percentage in enumerate(percentages):
        print(f"Digit {digit}: {percentage:.2f}%")

    best_digit = np.argmax(percentages)
    print(f"\nPredicted digit: {best_digit}")
    print(f"Confidence: {percentages[best_digit]:.2f}%")



# Inputs range [0, 1] manually normalized
# Weight range [-0.5, 0.5] initial
# Output range [0, 1] sigmoid output

# Train cases
# Train cases
loader = DataLoader("data/10x10_thin_stroke_digits_1000.zip")
# loader = DataLoader("data/10x10_handwritten_digits_500.zip")
# loader = DataLoader("data/10x10_digit_training_dataset.zip")
cases = loader.load()

# training
epochs = 500
learning_rate = 0.1
modelName = "models/model_conv.npz"

# network = NeuralNetwork([
#     DenseLayer(100, 64),
#     DenseLayer(64, 32),
#     DenseLayer(32, 10)
# ])

network = NeuralNetwork(
    ModelStorage("models/model_conv.npz"),
    [
        ConvLayer(8,3), # outputs 512 8 filters 3 by 3
        DenseLayer(512, 32),
        DenseLayer(32, 10)
    ])

train_network(network, cases, epochs, learning_rate)

# test
inputs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
resh = np.asarray(inputs).reshape(10, 10)
prediction = network.forwardActivations(resh)[-1]

# visualize prediction
visualize_prediction(resh, network)

# save model
network.save()
print(f"Model saved to {modelName}")