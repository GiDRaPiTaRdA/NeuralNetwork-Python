from abc import ABC, abstractmethod
import numpy as np
class ActivationFunction(ABC):
    @abstractmethod
    def function(self, output: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, output: np.ndarray) -> np.ndarray:
        pass