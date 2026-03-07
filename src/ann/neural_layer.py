"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""
import numpy as np 

class NeuralLayer:
    def __init__(self,input_dim,output_dim, weight_init='xavier'):
        # we store dimensions 
        self.input_dim = input_dim
        self.output_dim = output_dim
        # we initialise weights here
        # Implementation of xavier and random
        if weight_init == 'xavier':
            limit = np.sqrt(6 / (input_dim + output_dim))
            self.W = np.random.uniform(-limit, limit, (input_dim, output_dim))
        else:
            self.W = np.random.randn(input_dim, output_dim) * 0.01
        # initialise bias here
        self.b = np.zeros((1,output_dim))

        self.grad_W = None
        self.grad_b = None

    def forward(self,X):
        # we store the input for the backward pass
        self.X = X
        # Z = X.W + b
        Z = X @ self.W + self.b
        return Z
    def backward(self,dZ):
        # dZ is the gradient with respect to the linear output Z
        # self.grad_W = dL/dW = X^T * dZ 
        batch_size = self.X.shape[0]
        # we compute gradients now
        self.grad_W = (self.X.T @ dZ) / batch_size
        self.grad_b = np.sum(dZ, axis=0, keepdims=True) / batch_size
        # gradient to pass backward
        dX = dZ @ self.W.T
        return dX