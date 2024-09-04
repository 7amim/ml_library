"""
Steps

1. Create a bootstrapped dataset in which the same sample can appear twice in the bootstrapped dataset, as long as it
is the same length as the original dataset

2. Create a decision tree using the bootstrapped dataset - specifically a random subset of a certain length
 of the dataset

3. Do step 1 and 2 many times

4. We can use the consensus of the decision trees to predict a class

5. We use the out-of-bag error, calculated on the samples that did not enter the decision tree consideration,
to evaluate the accuracy of the forest

6. Then we can do step 1 and 2 for a new number of variables, i.e. create a random forest again, redo steps 1-5 and
compare the out-of-bag error against the previous out-of-bag error
"""
from lib.decision_tree.decision_tree import  DecisionTree
from scipy.stats import mode

import numpy as np

class RandomForest:
    def __init__(self, n_estimators: int = 100, n_features: int = 2, max_depth: int = 100, min_samples_split: int = 2,
                 random_state: int = 42):
        self.n_estimators = n_estimators
        self.n_features = n_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split\
        self.random_state = random_state

        # Setting random seed for reproducibility
        if random_state is not None:
            np.random.seed(random_state)

        self.trees = []

    def fit(self, features, target, label):
        for i in range(self.n_estimators):
            bootstrapped_features, bootstrapped_target = RandomForest.bootstrap_features(features, target)
            selected_features = RandomForest.select_features(bootstrapped_features, self.n_features)
            current_decision_tree = DecisionTree(selected_features, bootstrapped_target, label)
            current_decision_tree.fit()
            self.trees.append(current_decision_tree)

    @staticmethod
    def bootstrap_features (features, target):
        rows, cols = features.shape
        bootstrap_index = np.random.choice(rows, rows, replace=True)
        return features[bootstrap_index], target[bootstrap_index]

    @staticmethod
    def select_features(features, n_features):
        cols = features.shape[1]
        feature_indices = np.random.choice(cols, n_features, replace=False)  # Randomly select column indices
        return features[:, feature_indices]

    def predict(self, features):
        predictions = np.array([tree.predict(features) for tree in self.trees])
        return mode(predictions, axis=0).mode[0]
