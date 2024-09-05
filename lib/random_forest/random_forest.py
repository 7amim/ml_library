from lib.decision_tree.decision_tree import  DecisionTree
from scipy.stats import mode

import numpy as np

class RandomForest:
    def __init__(self,
                 n_estimators: int = 100,
                 n_features: int = 2,
                 max_depth: int = 100,
                 min_samples_split: int = 2,
                 random_state: int = 42):
        """
        Creates a RandomForest classifier that is based on the DecisionTree class.

        :param n_estimators: the number of trees in the forest
        :param n_features: the number of features to select from for each tree
        :param max_depth: the maximum depth of the tree
        :param min_samples_split: the minimum number of samples required to split a node
        :param random_state: used to reproduce the results
        """
        self.n_estimators = n_estimators
        self.n_features = n_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state

        # Setting random seed for reproducibility
        if random_state is not None:
            np.random.seed(random_state)

        self.trees = []

    def fit(self,
            features,
            target,
            label=1):
        """
        Given the features and target labels, fits the trees to the data.

        :param features: the training feature set
        :param target: the training target set
        :param label: the target label
        """
        for i in range(self.n_estimators):
            bootstrapped_features, bootstrapped_target = RandomForest.bootstrap_features(features, target)
            selected_features = RandomForest.select_features(bootstrapped_features, self.n_features)
            current_decision_tree = DecisionTree(selected_features, bootstrapped_target, label)
            current_decision_tree.fit()
            self.trees.append(current_decision_tree)

    @staticmethod
    def bootstrap_features(features,
                           target):
        """
        Returns the bootstrapped dataset.

        :param features: the original training feature set
        :param target: the original target set
        :return: the bootstrapped dataset with its labels
        """
        rows, cols = features.shape
        bootstrap_index = np.random.choice(rows, rows, replace=True)
        return features[bootstrap_index], target[bootstrap_index]

    @staticmethod
    def select_features(features,
                        n_features):
        """
        Selects the number of features to use when generating a given tree.

        :param features: the training feature set
        :param n_features: the number of features to choose from
        :return:
        """
        cols = features.shape[1]
        feature_indices = np.random.choice(cols, n_features, replace=False)  # Randomly select column indices
        return features[:, feature_indices]

    def predict(self,
                features):
        """
        Returns the predictions for a given set of features.

        :param features: the features to perform prediction on
        :return: the labels for each input feature
        """
        predictions = np.array([tree.predict(features) for tree in self.trees])
        return mode(predictions, axis=0).mode[0]
