from activationFunctions.abstractfunction import ActivationFunction
import numpy as np

class ReLUActivation(ActivationFunction):
    def function(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def derivative(self, output: np.ndarray) -> np.ndarray:
        return np.where(output > 0, 1, 0)