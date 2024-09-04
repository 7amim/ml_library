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