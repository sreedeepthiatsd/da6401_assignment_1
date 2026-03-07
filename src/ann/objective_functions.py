"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""

import numpy as np


class CrossEntropyLoss:
    def forward(self, logits, y_true):
        """
        logits: (batch_size, num_classes)
        y_true: one-hot encoded labels (batch_size, num_classes)
        """

        # numerical stability
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.probs = exp / np.sum(exp, axis=1, keepdims=True)

        self.y_true = y_true

        # compute loss (mean)
        loss = -np.mean(np.sum(y_true * np.log(self.probs + 1e-9), axis=1))
        return loss

    def backward(self):
        """
        derivative of softmax + cross entropy simplifies to:
        dZ = (probs - y_true) / batch_size
        """
        batch_size = self.y_true.shape[0]
        dZ = (self.probs - self.y_true) / batch_size
        return dZ
class MSE:
    def forward(self, y_pred, y_true):
        """
        y_pred:  Output of the last layer (batch_size, num_classes) 
        y_true:  One-hot labels (batch_size, num_classes) 
        """
        self.y_pred = y_pred
        self.y_true = y_true
        # calculate mean square error across all samples and classes
        loss = np.mean(np.square(self.y_pred - self.y_true))
        return loss
    def backward(self):
        """
        The gradient of MSE: 2/n * (y_pred - y_true)
        """
        batch_size = self.y_true.shape[0]
        num_classes = self.y_true.shape[1]
        dA = 2.0 * (self.y_pred - self.y_true) / (batch_size * num_classes)
        return dA
        