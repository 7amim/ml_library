**Decision Tree Steps:**

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

***Returns***:
A decision tree that can be used for classification or regression tasks, depending on the target variable type.
