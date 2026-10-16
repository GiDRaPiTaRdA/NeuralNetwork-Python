import numpy as np
from numpy.typing import NDArray


from abc import ABC, abstractmethod


class ILayer(ABC):

    @abstractmethod
    def activate(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        pass

    @abstractmethod
    def backward(
        self,
        error: NDArray[np.float64],
        output: NDArray[np.float64],
        previous: NDArray[np.float64],
        learning_rate: float
    ) -> NDArray[np.float64]:
        pass

    def save(self) -> dict[str, np.ndarray]:
        return {}

    def load(self, data: dict[str, np.ndarray]) -> None:
        pass