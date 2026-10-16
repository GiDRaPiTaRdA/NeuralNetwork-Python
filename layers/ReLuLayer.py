# import numpy as np
# from numpy.typing import NDArray
# from activationFunctions.relu_function import ReLUActivation

# from .ILayer import ILayer


# class ReLULayer(ILayer):

#     def __init__(self):
#         self.activationfunction = ReLUActivation()

#     def activate(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
#         return self.activationfunction.function(inputs)

#     def backward(
#         self,
#         error: NDArray[np.float64],
#         output: NDArray[np.float64],
#         previous: NDArray[np.float64],
#         learning_rate: float
#     ) -> NDArray[np.float64]:
#         return error * (previous > 0)