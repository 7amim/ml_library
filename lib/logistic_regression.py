
import numpy as np

class LogisticRegression():

    def __init__(self,
                 learning_rate: int = 0.001,
                 iterations: int = 100):
        """
        Initializes the logistic regression model.

        :param learning_rate: the rate at which the weights should be adjusted during gradient descent
        :param iterations: the number of iterations of the gradient descent algorithm
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.cost_history = []

    @staticmethod
    def sigmoid(z):
        """
        Computes the sigmoid function.

        :param z: the linear combination of features and weights
        :return: the probability using the sigmoid function
        """
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def get_cost(probabilities: np.array,
                 targets: np.array):
        """
        Computes the log-liklihood loss

        :param probabilities: the probability vector
        :param targets: the target labels
        :return: the computed cost
        """
        return -np.sum(targets * np.log(probabilities) + (1 - targets) * np.log(1 - probabilities)) / len(targets)


    def fit(self, features, targets):
        """
        Trains the model using gradient descent.

        :param features: the feature matrix to train from
        :param targets: the labels to compare against during training
        """
        rows = features.shape[0]
        self.weights = np.random.rand(rows, 1)
        self.cost_history = []

        for i in range(self.iterations):

            z = np.dot(features.T, self.weights)
            probabilities = LogisticRegression.sigmoid(z)

            gradient = np.dot(features.T, (targets - probabilities)) / len(targets)

            cost = LogisticRegression.get_cost(probabilities, targets)
            self.cost_history.append(cost)

            self.weights -= self.learning_rate * gradient

    def predict(self, features: np.array):
        """
        Given a new set of features, predicts the label the features belong to.

        :param features: the new set of features to predict a label for
        :return: the predicted labels
        """
        z = np.dot(features, self.weights)
        probabilities = self.sigmoid(z)

        return [1 if p > 0.5 else 0 for p in probabilities]

    def predict_probabilities(self, features: np.array):
        """
        Given a new set of features, predicts the probabilities for each feature

        :param features: the new set of features to predict a probability for
        :return: the predicted probabilities
        """
        z = np.dot(features, self.weights)
        probabilities = self.sigmoid(z)

        return probabilities
