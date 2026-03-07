"""
Optimization Algorithms
Implements: SGD, Momentum,NAG,RMS prop etc.
"""
import numpy as np

class SGD:
    def __init__(self,learning_rate):
        self.lr = learning_rate
    
    def update(self, layer):
        # we have to update weights and biases using the stored gradients
        layer.W -= self.lr * layer.grad_W
        layer.b -= self.lr * layer.grad_b

class Momentum:
    def __init__(self, learning_rate, momentum=0.9):
        self.lr = learning_rate
        self.gamma = momentum
        self.v_W = {}
        self.v_b = {}
    def update(self,layer):
        if layer not in self.v_W:
            self.v_W[layer] = np.zeros_like(layer.grad_W)
            self.v_b[layer] = np.zeros_like(layer.grad_b)

        self.v_W[layer] = self.gamma * self.v_W[layer] + self.lr * layer.grad_W
        self.v_b[layer] = self.gamma * self.v_b[layer] + self.lr * layer.grad_b
        layer.W -=  self.v_W[layer]
        layer.b -= self.v_b[layer]

class NAG:
    def __init__(self, learning_rate,momentum = 0.9):
        self.lr = learning_rate
        self.gamma = momentum
        self.v_W = {}
        self.v_b = {}
    def update(self,layer):
        if layer not in self.v_W:
            self.v_W[layer] = np.zeros_like(layer.grad_W)
            self.v_b[layer] = np.zeros_like(layer.grad_b)

        # we have to store previous velocity
        v_W_prev = self.v_W[layer]
        v_b_prev = self.v_b[layer]
        # we have to update velocity
        self.v_W[layer] = self.gamma * self.v_W[layer] + self.lr * layer.grad_W
        self.v_b[layer] = self.gamma * self.v_b[layer] + self.lr * layer.grad_b
        # Look-ahead update logic
        layer.W -= (self.gamma * v_W_prev + (1 + self.gamma) * self.v_W[layer])
        layer.b -= (self.gamma * v_b_prev + (1 + self.gamma) * self.v_b[layer])

class RMSProp:
    def __init__(self,learning_rate, beta=0.9, epsilon=1e-8):
        self.lr = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.v_W = {}
        self.v_b = {}
    def update(self, layer):
        if layer not in self.v_W:
            self.v_W[layer] = np.zeros_like(layer.grad_W)
            self.v_b[layer] = np.zeros_like(layer.grad_b)
        # we have to update running average of squared gradients
        self.v_W[layer] = self.beta * self.v_W[layer] + (1-self.beta)*(layer.grad_W**2)
        self.v_b[layer] = self.beta * self.v_b[layer] + (1-self.beta)*(layer.grad_b**2)
        # update parameters with adaptive scaling
        layer.W -= (self.lr / (np.sqrt(self.v_W[layer]) + self.epsilon)) * layer.grad_W
        layer.b -= (self.lr / (np.sqrt(self.v_b[layer]) + self.epsilon)) * layer.grad_b

