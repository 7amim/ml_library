import torch
from torch import nn, optim, arange

from neural_network import NeuralNetwork
from sklearn import datasets
from sklearn.model_selection import train_test_split

import copy
import tqdm


def train(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    batch_start = arange(0, len(X_train), batch_size)

    best_validation_accuracy = 0
    best_model_weights = None

    for epoch in tqdm.tqdm(range(epochs), desc="Training Epochs"):
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=False) as bar:
            bar.set_description(f"Epoch {epoch + 1}")

            for i in bar:
                # Get the current batch
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size].float()

                # Perform the forward pass
                y_pred = model(X_batch)
                loss = loss_function(y_pred.view(-1, 1), y_batch.view(-1, 1))

                # Perform the backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Calculate accuracy for the current batch
                predictions = (y_pred > 0.5).float()
                accuracy = (predictions == y_batch.view(-1, 1)).float().mean()

                bar.set_postfix(loss=float(loss), accuracy=float(accuracy))

        # Evaluate at each epoch
        model.eval()
        with torch.no_grad():
            y_pred_validation = model(X_val)
            predictions_val = (y_pred_validation > 0.5).float()
            accuracy_validation = (predictions_val == y_val.view(-1, 1)).float().mean()

        if accuracy_validation > best_validation_accuracy:
            best_validation_accuracy = accuracy_validation
            best_model_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_weights)
    return best_validation_accuracy

X, y = datasets.load_breast_cancer(return_X_y=True)

# Get the train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# Get the validation set from the train set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

# Convert data to appropriate tensor types
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)

X_val = torch.FloatTensor(X_val)
y_val = torch.FloatTensor(y_val)

X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

# Initialize neural network
neural_network = NeuralNetwork()

# Train the model
validation_accuracy = train(neural_network, X_train, y_train, X_val, y_val)

print(f"Best validation accuracy: {validation_accuracy:.4f}")

# Test the model
y_test_pred = neural_network(X_test)
predictions = (y_test_pred > 0.5).float()
test_accuracy = (predictions == y_test.view(-1, 1)).float().mean()

print(f"Best test accuracy: {test_accuracy:.4f}")
