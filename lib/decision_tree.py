"""
Builds a decision tree from a dataset by recursively splitting the data to minimize impurity.

Decision trees are constructed to predict a target label by recursively partitioning the data based
on features that best separate the target labels. The goal is to create a tree where each leaf node
represents a pure subset of the data (i.e., all instances in a leaf have the same target label).

Steps:

1. **Categorical Variables:**
   - For each categorical feature, count the occurrences of each unique value with respect to the target label (e.g., count "Yes" and "No" for each value).
   - Calculate the probability of the target label ("Yes" or "No") given each unique value of the feature.
   - Compute the Gini impurity or entropy for the feature based on these probabilities:
     - Gini impurity is calculated as:
       Gini = 1 - sum(p_i^2), where p_i is the probability of each class in the node.
   - The total Gini impurity of a split is the weighted sum of the impurity of each resulting subset (leaf), weighted by the number of samples in each subset.

2. **Numerical Variables:**
   - For each numerical feature, sort the data by the feature's values.
   - Evaluate potential split points, typically the midpoints between consecutive values.
   - Calculate the impurity (Gini impurity or entropy) for each potential split.
   - Select the split point that results in the lowest impurity.

3. **Selecting the Best Feature:**
   - Choose the feature (categorical or numerical) that yields the lowest impurity (Gini or entropy) as the root node of the decision tree.
   - Repeat the process recursively for each subset of data created by the split, adding nodes to the tree.

4. **Recursive Splitting:**
   - Continue splitting subsets until a stopping criterion is met, such as:
     - Maximum tree depth.
     - Minimum number of samples per leaf.
     - No further reduction in impurity is possible.

5. **Handling Overfitting:**
   - To prevent overfitting, use techniques like pruning (removing nodes that provide minimal predictive power), setting a maximum depth, or requiring a minimum number of samples per leaf.

6. **Metrics for Splitting:**
   - Decision trees can use different metrics (e.g., Gini impurity or entropy) to determine the best split, depending on the problem context and specific implementation.

Returns:
A decision tree that can be used for classification or regression tasks, depending on the target variable type.
"""

import numpy as np

class Node:

    def __init__(self,
                 feature: list | np.array,
                 target: list | np.array,
                 label: int,
                 feature_idx: int):

        self.feature: np.array = feature
        self.categories: set = set(self.feature)
        self.target: np.array = target
        self.label = label

        self.total_samples = len(self.feature[~np.isnan(self.feature)])

        self.impurity = None
        self.depth = None  # Not organized in tree, so no depth

        self.branches = {}

    @staticmethod
    def gini_impurity(num_yes: int, num_no: int) -> float:
        if num_yes + num_no == 0:
            return 0  # Avoid division by zero
        return 1 - np.square(num_yes / (num_yes + num_no)) - np.square(num_no / (num_yes + num_no))

    def weighted_impurity(self):
        """
        Sets the total impurity for node based on the weighted sum of the impurity of each
        category.
        """
        weighted_impurity = 0

        self.category_impurity = {}

        for c in self.categories:
            # Determine the number of yes and no votes for each category
            num_yes = np.sum((self.feature == c) & (self.target == self.label))
            num_no = np.sum((self.feature == c) & (self.target != self.label))

            if num_yes + num_no == 0:
                continue

            num_c = num_yes + num_no

            c_impurity = Node.gini_impurity(num_yes, num_no)

            # Store for future use when creating splits in a decision tree
            self.category_impurity[c] = c_impurity

            # Calculate the impurity associated with each category and weight it by its proportion in the feature
            weighted_impurity += (num_c / len(self.feature)) * c_impurity

        self.impurity = weighted_impurity

    def set_depth(self, depth: int):
        self.depth = depth

    def get_majority_class(self):
        """
        Calculates the majority class of the target at this node

        :return: the label for the majority class
        """
        unique, counts = np.unique(self.target, return_counts=True)
        majority_class = unique[np.argmax(counts)]
        return majority_class


class DecisionTree:

    def __init__(self, features, target, label, feature_names=None, max_depth: int = 5):
        self.root = None

        self.features = features
        self.target = target
        self.label = label

        self.feature_names = feature_names if feature_names else np.arange(len(self.features))

        self.feature_nodes = {}

        self.max_depth = max_depth

    def determine_root(self):
        # Find the root node by identifying the node with the minimum impurity
        min_impurity = 1
        root = None

        for idx, feature in enumerate(self.features.T):
            feature_node = Node(feature, self.target, self.label, idx)
            feature_node.weighted_impurity()

            if feature_node.impurity < min_impurity:
                root = feature_node
                min_impurity = feature_node.impurity

        root.set_depth(1)
        return root

    def fit(self):
        self.root = self.determine_root()
        self.split_node(self.root)

    def split_node(self, node):
        if node.depth >= self.max_depth:
            return

        for c in node.categories:
            if node.category_impurity[c] > 0:
                # Find the indices of where the current category is active
                subset_indices = np.where(self.features[:, node.feature_idx] == c)
                subset_target = self.target[subset_indices]

                # Index the remaining features based on the category
                remaining_features = self.features[subset_indices]

                min_impurity = 1
                best_child_node = None

                # Traverse the other features to find the next feature node based on minimum impurity
                for idx, feature in enumerate(remaining_features.T):
                    child_node = Node(feature, subset_target, self.label, idx)
                    child_node.weighted_impurity()

                    if child_node.impurity < min_impurity:
                        min_impurity = child_node.impurity
                        best_child_node = child_node

                if best_child_node:
                    best_child_node.set_depth(node.depth + 1)
                    node.branches[c] = best_child_node

                    # Recursively split the child node
                    self.split_node(best_child_node)

    def predict(self, samples):
        if len(samples.shape) == 1:
            return self.traverse_tree(samples, self.root)
        else:  # More than one sample
            return [self.traverse_tree(s, self.root) for s in samples]

    def traverse_tree(self, sample, node):
        if not node.branches:  # Arrived at a leaf node
            return node.get_majority_class()

        # Get the feature value of the splitting feature for current node
        feature_value = sample[node.feature_idx]

        # Continue to traverse
        if feature_value in node.branches:
            return self.traverse_tree(sample, node.branches[feature_value])
        else:
            # If the feature value doesn't match any branch, return the majority class
            return node.get_majority_class()
