from flask import Flask, request, jsonify, send_file
from neural_network import NeuralNetwork
import numpy as np

app = Flask(__name__)

network = NeuralNetwork([100, 64, 32, 10])
network.load("models/model.npz")


@app.route("/")
def index():
    return send_file("pixel_drawer.html")


@app.route("/guess", methods=["POST"])
def guess():
    data = request.get_json()
    pixels = data.get("pixels")

    if not isinstance(pixels, list) or len(pixels) != 100:
        return jsonify({"error": "Expected 100 pixels"}), 400

    outputs = network.forwardActivations(pixels)[-1]

    return jsonify({
        "prediction": int(np.argmax(outputs)),
        "scores": (outputs * 100).tolist()
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)