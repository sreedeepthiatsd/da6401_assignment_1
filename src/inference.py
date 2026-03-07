"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
import numpy as np
from src.utils.data_loader import load_data
from src.ann.neural_network import NeuralNetwork

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def parse_arguments():
    """
    Parse command-line arguments for inference.
    
    TODO: Implement argparse with:
    - model_path: Path to saved model weights(do not give absolute path, rather provide relative path)
    - dataset: Dataset to evaluate on
    - batch_size: Batch size for inference
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    """
    parser = argparse.ArgumentParser(description='Run inference on test set')
    parser.add_argument(
        "--model_path",
        type=str,
        default="src/best_model.npy",
        help="Path to saved model weights"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mnist", "fashion_mnist"],
        default="mnist"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        nargs='+',
        default=[128,128,128]
    )

    parser.add_argument(
        "--activation",
        type=str,
        choices=["relu","sigmoid","tanh"],
        default="tanh"
    )
    
    parser.add_argument(
        "--loss",
        type=str,
        choices=["cross_entropy","mse"],
        default="cross_entropy"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="rmsprop"
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001
    )

    parser.add_argument(
        "--weight_init",
        type=str,
        default="xavier"
    )

    
    return parser.parse_args()


def load_model(model_path):
    """
    Load trained model from disk.
    """
    data = np.load(model_path, allow_pickle=True).item()
    return data


def evaluate_model(model, X_test, y_test): 
    """
    Evaluate model on test data.
        
    TODO: Return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    logits = model.forward(X_test)

    predictions = np.argmax(logits, axis=1)
    true_labels = np.argmax(y_test, axis=1)
    loss = model.loss_fn.forward(logits,y_test)
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average="macro")
    recall = recall_score(true_labels, predictions, average="macro")
    f1 = f1_score(true_labels, predictions, average="macro")

    results = {
        "logits": logits,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    return results


def main():
    """
    Main inference function.

    TODO: Must return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    args = parse_arguments()
    print("Loading dataset...")
    _, _, x_test, y_test = load_data(args.dataset)

    print("Creating neural network...")
    model = NeuralNetwork(args)

    print("Loading trained weights...")
    weights = load_model(args.model_path)

    model.set_weights(weights["weights"])

    print("Evaluating model...")
    results = evaluate_model(model, x_test, y_test)

    print("Accuracy:", results["accuracy"])
    print("Precision:", results["precision"])
    print("Recall:", results["recall"])
    print("F1 Score:", results["f1"])
    
    print("Evaluation complete!")
    return results


if __name__ == '__main__':
    main()
