from activationFunctions.abstractfunction import ActivationFunction
import numpy as np

class SigmoidActivation(ActivationFunction):
    def function(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def derivative(self, output: np.ndarray) -> np.ndarray:
        return output * (1 - output)