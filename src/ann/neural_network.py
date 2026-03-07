"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import numpy as np
from src.ann.neural_layer import NeuralLayer
from src.ann.activations import ReLU, Sigmoid,Tanh
from src.ann.objective_functions import CrossEntropyLoss, MSE
from src.ann.optimizers import SGD, Momentum, NAG, RMSProp

class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """
    
    def __init__(self, cli_args):
        """
        Initialize the neural network.

        Args:
            cli_args: Command-line arguments for configuring the network
        """
        # Basic configuration from CLI
        input_dim = 784
        output_dim = 10
        
        hidden_sizes = cli_args.hidden_size  # assume list
        self.learning_rate = cli_args.learning_rate

        # Selection of Optimizer
        if cli_args.optimizer == "sgd":
            self.optimizer = SGD(self.learning_rate)
        elif cli_args.optimizer == "momentum":
            self.optimizer = Momentum(self.learning_rate)
        elif cli_args.optimizer == "nag":
            self.optimizer = NAG(self.learning_rate)
        elif cli_args.optimizer == "rmsprop":
            self.optimizer = RMSProp(self.learning_rate)
        else:
            raise ValueError("Unsupported optimizer")
        
        # Build layers
        self.layers = []
        prev_dim = input_dim
        
        for size in hidden_sizes:
            self.layers.append(NeuralLayer(prev_dim, size, weight_init=cli_args.weight_init))
            # Choosing activation function
            if cli_args.activation == "relu":
                activation = ReLU()
            elif cli_args.activation == "sigmoid":
                activation = Sigmoid()
            elif cli_args.activation == "tanh":
                activation = Tanh()
            else:
                raise ValueError("Unsupported loss function")

            self.layers.append(activation)
            prev_dim = size
        
        # Output layer (no activation)
        self.layers.append(NeuralLayer(prev_dim, output_dim, weight_init=cli_args.weight_init))
        if cli_args.loss == "cross_entropy":
            self.loss_fn = CrossEntropyLoss()
        elif cli_args.loss == "mse":
            self.loss_fn = MSE()
        else:
            raise ValueError("Unsupported loss function")
    
    
    def forward(self, X):
        """
        Forward propagation through all layers.
        
        Args:
            X: Input data
            
        Returns:
            Output logits
        """
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.
        
        Args:
            y_true: True labels
            y_pred: Predicted outputs
            
        Returns:
            return grad_w, grad_b
        """
        loss = self.loss_fn.forward(y_pred, y_true)
        dZ = self.loss_fn.backward()
        
        for layer in reversed(self.layers):
            dZ = layer.backward(dZ)
        
        return loss
    
    def update_weights(self):
        """
        Update weights using the selected optimizer.
        """
        for layer in self.layers:
            if hasattr(layer, "W"):
                self.optimizer.update(layer)
    
    def train(self, X_train, y_train, epochs, batch_size):
        """
        Train the network for specified epochs.
        """
        num_samples = X_train.shape[0]
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(0, num_samples, batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                
                logits = self.forward(X_batch)
                loss = self.backward(y_batch, logits)
                self.update_weights()
                epoch_loss += loss
            #losses.append(loss)
            epoch_loss /= (num_samples // batch_size)
            losses.append(epoch_loss)
        return losses
            
            
    def get_weights(self):

        weights = []

        for layer in self.layers:
            if hasattr(layer, "W"):
                weights.append({
                "W": layer.W,
                "b": layer.b
                })
        return weights
    def set_weights(self, weights):

        idx = 0

        for layer in self.layers:
            if hasattr(layer, "W"):
                layer.W = weights[idx]["W"]
                layer.b = weights[idx]["b"]
                idx += 1
    
    def evaluate(self, X, y):
        """
        Evaluate the network on given data.
        """
        logits = self.forward(X)
        
        predictions = np.argmax(logits, axis=1)
        true_labels = np.argmax(y, axis=1)
        
        accuracy = np.mean(predictions == true_labels)
        
        return accuracy
