import torch
from torch import nn, optim, arange

from neural_network import NeuralNetwork
from sklearn import datasets
from sklearn.model_selection import train_test_split

import copy
import tqdm

def train(model, X_train, y_train, X_val, y_val, epochs=50, batch_size = 32):
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    batch_start = arange(0, len(X_train), batch_size)

    best_validation_accuracy = 0
    best_model_weights = None

    for epoch in range(epochs):
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch+1}")

            for i in bar:
                # Get the current batch
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                # Perform the forward pass
                y_pred = model(X_batch)
                loss = loss_function(y_pred, y_batch)
                # Perform the backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                accuracy = (y_pred == y_batch).float().mean()

                bar.set_postfix(loss=float(loss),
                                accuracy=float(accuracy))
        model.eval() # evaluates at each epoch
        y_pred_validation = model(X_val)
        accuracy_validation = (y_pred_validation == y_val).float().mean()

        best_validation_accuracy = max(accuracy_validation, best_validation_accuracy)
        best_model_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_weights)
    return best_validation_accuracy

X, y = datasets.load_breast_cancer(return_X_y=True)

# Get the train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# Get the validation set from the train set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)  # 0.25 x

X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)

X_val = torch.tensor(X_val)
y_val = torch.tensor(y_val)

neural_network = NeuralNetwork()

train(neural_network, X_train, y_train, X_val, y_val)
