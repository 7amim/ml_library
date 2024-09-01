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
    def __init__(self, features, target, label, feature_names=None, max_depth: int = 100):
        """
        Builds a decision tree from a dataset by recursively splitting the data to minimize impurity.

        Decision trees are constructed to predict a target label by recursively partitioning the data based
        on features that best separate the target labels. The goal is to create a tree where each leaf node
        represents a pure subset of the data (i.e., all instances in a leaf have the same target label), or to
        minimize the impurity of each leaf node (i.e., the amount of mixture of target labels.)

        :param features: the feature matrix to be fit to the tree
        :param target: the training labels used to fit the tree
        :param label: the target label to be considered the positive class
        :param feature_names: the names of the features in the feature column
        :param max_depth: the maximum depth of the tree before ending algorithm
        """
        self.root = None

        self.features = features
        self.target = target
        self.label = label

        self.feature_names = feature_names if feature_names else np.arange(len(self.features))

        self.feature_nodes = {}

        self.max_depth = max_depth

    def determine_root(self):
        """
        Given the features, determines which feature should be used for the root node based on identifying
        which feature results in the minimum impurity.

        :return: the root node of the tree
        """
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
        """
        Determines the root node, and then builds the decision tree by identifying which feature
        splits are necessary.
        """
        self.root = self.determine_root()
        self.split_node(self.root)

    def split_node(self, node):
        """
        Splits a given node into child nodes based on the feature with the lowest impurity.

        For a given node:
        - Identifies the categories within the node's feature set.
        - Finds the subset of data corresponding to each category for the current feature.
        - Evaluates the remaining features to find the feature with the lowest impurity
          for the current subset of data.
        - Creates a child node for each category using the feature with the minimum impurity.
        - Recursively applies the same splitting logic to each child node until a stopping
          criterion is met (e.g., maximum depth, minimum impurity decrease).

        :param node: Node object representing the current node to be split.
        """
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
        """
        Given a set of samples, traverses the decision tree to classify the sample.

        :param samples: Can either be a list of features or a single set of features to predict
        :return: the class prediction
        """
        if len(samples.shape) == 1:
            return self.traverse_tree(samples, self.root)
        else:  # More than one sample
            return [self.traverse_tree(s, self.root) for s in samples]

    def traverse_tree(self, sample, node):
        """
        Given a set of features, and the current node, recursively traverses the tree until it reaches a
        leaf node. Once a leaf node is found, makes a prediction based on the majority class
        that corresponds to the node.

        :param sample: the set of features to make a prediction for
        :param node: the current node to be traversed
        :return: returns the predicted class of the sample
        """
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
