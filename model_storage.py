import numpy as np


class ModelStorage:
    def __init__(self, path="model.npz"):
        self.path = path

    def save(self, layers):
        data = {}

        for i, layer in enumerate(layers):
            for name, value in layer.save().items():
                data[f"{i}_{name}"] = value

        np.savez(self.path, **data)

    def load(self, layers):
        with np.load(self.path) as data:
            for i, layer in enumerate(layers):
                layer_data = self.get_layer_data(data, i)
                layer.load(layer_data)

    def get_layer_data(self, data, index):
        prefix = f"{index}_"
        layer_data = {}

        for key in data.files:
            if key.startswith(prefix):
                name = key.removeprefix(prefix)
                layer_data[name] = data[key]

        return layer_data