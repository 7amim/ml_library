from random_forest import RandomForest
from sklearn import datasets
from sklearn.model_selection import train_test_split

import numpy as np

def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

# This dataset has binary labels
dataset = datasets.load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(dataset.data,
                                                    dataset.target,
                                                    test_size=0.2,
                                                    random_state=1234)

classifier = RandomForest()
classifier.fit(X_train, y_train, 1)

predictions = classifier.predict(X_test)

accuracy = accuracy(y_test, predictions)

print(accuracy)
