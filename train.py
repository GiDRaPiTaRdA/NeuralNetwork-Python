"""Simple matrix multiplication example. No neural network implementation."""
import numpy as np
import ast
from neural_network import NeuralNetwork, Layer, TrainCase
from data.data_loader import DataLoader

def train_network(network, cases, epochs, learning_rate):
    for epoch in range(epochs):
        total_error = 0

        for case in cases:
            error = network.train(
                case.inputs,
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
modelName = "models/model.npz"

network = NeuralNetwork([100,64,32,10])

train_network(network, cases, epochs, learning_rate)

# test
inputs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
prediction = network.forwardActivations(inputs)[-1]

# visualize prediction
visualize_prediction(inputs, network)

# save model
network.save(modelName)
print(f"Model saved to {modelName}")