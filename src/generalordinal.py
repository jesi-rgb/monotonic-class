import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.datasets import load_iris




# subset splitting
def split_subsets(data, target):
    # figure out how many factors there are in the target value
    unique_target = np.unique(target)
    print(unique_target)


# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)
# clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth=3, min_samples_leaf=5)
# clf_entropy.fit(X_train, y_train)
# probs = clf_entropy.predict_proba(X_test)


if __name__ == "__main__":
    data = load_iris().data
    target = load_iris().target

    split_subsets(data, target)