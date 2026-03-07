"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""
from sklearn.metrics import precision_score, recall_score, f1_score
import json
import wandb
from src.utils.data_loader import load_data
from src.ann.neural_network import NeuralNetwork
import numpy as np
import argparse

def parse_arguments():
    """
    Parse command-line arguments.
    
    TODO: Implement argparse with the following arguments:
    - dataset: 'mnist' or 'fashion_mnist'
    - epochs: Number of training epochs
    - batch_size: Mini-batch size
    - learning_rate: Learning rate for optimizer
    - optimizer: 'sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    - loss: Loss function ('cross_entropy', 'mse')
    - weight_init: Weight initialization method
    - wandb_project: W&B project name
    - model_save_path: Path to save trained model (do not give absolute path, rather provide relative path)
    """
    parser = argparse.ArgumentParser(description='Train a neural network')
    # dataset
    parser.add_argument("-d", "--dataset", type = str, choices = ['mnist', 'fashion_mnist'], default = 'mnist',help="Dataset to use")
    # ==== training parameters ======
    # epochs
    parser.add_argument("-e", "--epochs", type = int, default = 10)
    # batch size
    parser.add_argument("-b", "--batch_size", type = int, default = 32)
    # ======================================
    # learning rate
    parser.add_argument("-lr", "--learning_rate", type = float, default = 0.001)
    # optimizer
    parser.add_argument("-o", "--optimizer", type = str, choices = ['sgd', 'momentum', 'nag', 'rmsprop'], default = 'rmsprop')
    # ========== Architecture ====================
    # list of hidden layer sizes
    parser.add_argument("-sz","--hidden_size", type = int,nargs='+', default=[128,128,128], help="List of hidden layer sizes" ) # nargs='+' allows -sz 128 64 32
    # number of neurons in hidden layers
    parser.add_argument("-nhl", "--num_layers", type = int, default = 3)
    # activation
    parser.add_argument("-a", "--activation", type = str, choices = ['relu','sigmoid','tanh'] ,default = 'tanh')
    # loss function
    parser.add_argument("-l","--loss", type = str, choices = ['cross_entropy','mse'], default = 'cross_entropy')
    # weight initialization
    parser.add_argument("-w_i", "--weight_init", type = str, choices = ['random', 'xavier'], default = 'xavier')
    # wandb_project
    #parser.add_argument()
    # model saving
    parser.add_argument("--model_save_path", type=str, default="models/best_model.npy")
    # Weights & Biases arguments
    parser.add_argument("-wp", "--wandb_project", type=str, default="dl_assignment_1", help="WandB project name")
    parser.add_argument("-we", "--wandb_entity", type=str, default="ee24d044-indian-institute-of-technology-madras", help="WandB entity name")
    return parser.parse_args()


def main():
    """
    Main training function.
    """
    best_f1 = 0
    args = parse_arguments()
    wandb.init(project=args.wandb_project,entity=args.wandb_entity, config=vars(args))

    print("Loading dataset....")
    x_train, y_train, x_test, y_test = load_data(args.dataset)

    # -------- Data Exploration Logging --------


    # print("Logging sample images to W&B...")

    # table = wandb.Table(columns=["class", "image"])

    # # convert one-hot labels to class indices
    # labels = np.argmax(y_train, axis=1)

    # for class_id in range(10):

    #     # find indices of this class
    #     class_indices = np.where(labels == class_id)[0]

    #     # take first 5 samples
    #     sample_indices = class_indices[:5]

    #     for idx in sample_indices:

    #         image = x_train[idx].reshape(28,28)

    #         table.add_data(
    #         class_id,
    #         wandb.Image(image)
    #     )

    # wandb.log({"sample_images": table})

    print("Creating Neural Network.....")
    model = NeuralNetwork(args)

    print("Start Training.....")
    
    num_samples = x_train.shape[0]
    for epoch in range(args.epochs):
        indices = np.random.permutation(num_samples)
        x_train = x_train[indices]
        y_train = y_train[indices]

        epoch_loss = 0

        for i in range(0, num_samples, args.batch_size):
            
            X_batch = x_train[i:i+args.batch_size]
            y_batch = y_train[i:i+args.batch_size]

            logits = model.forward(X_batch)
            loss = model.backward(y_batch, logits)
            model.update_weights()
            epoch_loss += loss
        epoch_loss /= (num_samples // args.batch_size)

        print("Evaluating Model....")
        logits = model.forward(x_test)

        predictions = np.argmax(logits, axis=1)
        true_labels = np.argmax(y_test, axis=1)
        #accuracy = model.evaluate(x_test, y_test)
        accuracy = np.mean(predictions == true_labels)
        precision = precision_score(true_labels, predictions, average='macro')
        recall = recall_score(true_labels, predictions, average='macro')
        f1 = f1_score(true_labels, predictions, average='macro')
        if f1 > best_f1:
            best_f1 = f1
            best_weights = model.get_weights()
            np.save("src/best_model.npy", {"weights":best_weights})
            config = vars(args)
            with open("src/best_config.json", "w") as f:
                json.dump(config, f, indent=4)
            print("New best model saved with F1:", best_f1)

        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)

        wandb.log({
                "epoch": epoch + 1,
                "loss": loss,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
                })

        # wandb.log({
        # "epoch": epoch + 1,
        # "loss": loss,
        # "test_accuracy": accuracy
        # })

        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss}, Test Accuracy: {accuracy}")
    
    print("Training complete!")
    wandb.finish()
    



if __name__ == '__main__':
    main()
