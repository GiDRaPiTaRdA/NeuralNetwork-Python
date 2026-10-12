"""Simple matrix multiplication example. No neural network implementation."""
import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[7, 8], [9, 10], [11, 12]])

C = A @ B

print("A:\n", A)
print("B:\n", B)
print("A @ B:\n", C)
