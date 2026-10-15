# import ast
# import numpy as np
# from neural_network import NeuralNetwork

# modelName = "models/model.npz"

# network = NeuralNetwork([512,64,32,10])
# network.load(modelName)

# def input_prediction(network):
#     while True:
#         text = input("Paste pixels (or 'exit'): ")

#         if text.lower() == "exit":
#             break

#         try:
#             # 1D 100
#             pixels = ast.literal_eval(text)

#             if not isinstance(pixels, list) or len(pixels) != 100:
#                 print("Expected 100 pixels")
#                 continue

#             # 10×10 → Convolution → ReLU → Flatten (512) → Dense (64 → 32 → 10, sigmoid)


#             # 2D 10 x 10
#             pixels = np.array(pixels).reshape(10, 10)

#             # Convolutional layer
#             filters = np.random.randn(8, 3, 3) * 0.1
#             biases = np.zeros(8)

#             # Multiply fliters on image
#             windows = np.lib.stride_tricks.sliding_window_view(pixels, (3, 3))
#             output = np.einsum("ijxy,fxy->fij", windows, filters)
#             output += biases[:, None, None]

#             # Relu, make 0 negative input
#             output = np.maximum(0, output)

#             # 8x8x8 flattened into 512
#             flattened = output.flatten()

#             outputs = network.forwardActivations(flattened)[-1]

#             for digit, score in enumerate(outputs):
#                 print(f"Digit {digit}: {score * 100:.2f}%")

#             print(f"Prediction: {np.argmax(outputs)}\n")

#         except (ValueError, SyntaxError, TypeError) as e:
#             print(f"Invalid input: {e}")

# input_prediction(network)