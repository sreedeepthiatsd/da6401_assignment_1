"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""
import numpy as np
from keras.datasets import mnist,fashion_mnist

def load_data(dataset = "mnist"):
    if dataset == "mnist":
        (x_train,y_train), (x_test,y_test) = mnist.load_data()
    
    elif dataset == "fashion_mnist":
        (x_train,y_train), (x_test,y_test) = fashion_mnist.load_data()
    else:
        raise ValueError("Dataset must be 'mnist' or 'fashion_mnist'")
    # Normalize pixel values
    # Convert from [0,255] -> [0,1]
    x_train = x_train.astype(np.float32) /255.0
    x_test = x_test.astype(np.float32) /255.0

    # Flatten images
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0],-1)

    # One-hot encode labels
    
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]

    return x_train, y_train, x_test, y_test
