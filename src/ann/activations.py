"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""
import numpy as np

class ReLU:
    def forward(self,Z):
        self.Z = Z
        A = np.maximum(0,Z)
        return A
    
    def backward(self,dA):
        # dA is the gradient of loss w.r.t to activation o/p
        # derivative of ReLU
        dZ = dA * (self.Z > 0)
        return dZ
class Sigmoid:
    def forward(self,Z):
        # self.Z = Z if we dont implement this, it can save memory
        self.A = 1 / (1 + np.exp(-Z))
        return self.A
    def backward(self,dA):
        # derivative is sigmoid(z)* (1 - sigmoid(z))
        return dA * (self.A *(1 - self.A))
class Tanh:
    def forward(self,Z):
        self.A = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
        return self.A
    def backward(self,dA):
        return dA * (1 - np.power(self.A, 2))