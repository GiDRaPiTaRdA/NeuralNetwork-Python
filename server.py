from flask import Flask, request, jsonify, send_file
from layers.Layer import DenseLayer
from neural_network import NeuralNetwork
from model_storage import ModelStorage
from layers.ConvLayer import ConvLayer
import numpy as np

app = Flask(__name__)

network = NeuralNetwork(
    ModelStorage("models/model_conv.npz"),
    [
        ConvLayer(8,3), # outputs 512 8 filters 3 by 3
        DenseLayer(512, 64),
        DenseLayer(64, 10)
    ])
network.load()


@app.route("/")
def index():
    return send_file("pixel_drawer.html")


@app.route("/guess", methods=["POST"])
def guess():
    data = request.get_json()
    pixels = data.get("pixels")

    if not isinstance(pixels, list) or len(pixels) != 100:
        return jsonify({"error": "Expected 100 pixels"}), 400

    resh = np.asarray(pixels).reshape(10, 10)
    outputs = network.forwardActivations(resh)[-1]

    return jsonify({
        "prediction": int(np.argmax(outputs)),
        "scores": (outputs * 100).tolist()
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)