import pandas as pd
import numpy as np
from decision_tree import DecisionTree 
from collections import Counter

# Original Data:
#  [[ 1  2  3]
#  [ 4  5  6]
#  [ 7  8  9]
#  [10 11 12]
#  [13 14 15]]

# Randomly Sampled Indices: [3 4 1 2 2]

# Resampled Data:
#  [[10 11 12]  # Sample 3
#   [13 14 15]  # Sample 4
#   [ 4  5  6]  # Sample 1
#   [ 7  8  9]  # Sample 2
#   [ 7  8  9]] # Sample 2 (Repeated)
def _most_common_label(y):
    counter = Counter(y)
    most_common = counter.most_common(1)[0][0]
    return most_common

def bootstrap_sample(X,y):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, size = n_samples, replace=True) #replace = true -> Sampling with replacement
    return X[idxs], y[idxs]

class RandomForest:
    def __init__(self, n_trees = 100, min_samples_split =2, max_depth = 100, n_feats = None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []   # To store each single tree that we are going to store

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(min_samples_split=self.min_samples_split, max_depth=self.max_depth, n_feats = self.n_feats)
            X_sample, y_sample = bootstrap_sample(X,y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    # Say you had 3 trees and 4 samples 
    # tree_preds = np.array([
    #     [0, 1, 1, 0],  # Tree 1 predictions
    #     [1, 1, 0, 0],  # Tree 2 predictions
    #     [0, 1, 1, 1]   # Tree 3 predictions
    # ])

    # we want something like (0,1,0) therefore 0 majority so 1st sample will be predicted as 0 
    # we have something like (1,1,1) therefore 1 majority so 2nd sample will be predicted as 1
    # we have something like (1,0,1) therefore 1 majority so 3rd sample will be predicted as 1
    # we have something like (0,0,1) therefore 0 majority so 4th sample will be predicted as 0
    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        # Majority vote 
        y_pred = [_most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)



# Testing
if __name__ == "__main__":
    # Imports
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    data = datasets.load_breast_cancer()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    clf = RandomForest(n_trees=3, max_depth=10)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy(y_test, y_pred)

    print("Accuracy:", acc)
