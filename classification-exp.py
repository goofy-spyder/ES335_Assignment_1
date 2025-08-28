import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score

np.random.seed(42)

# Code given in the question
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)
plt.scatter(X[:, 0], X[:, 1], c=y)

X_pd = pd.DataFrame(X)
y_pd = pd.Series(y)

split_point = int(len(X) * 0.70)
X_train = pd.DataFrame(X[split_point:])
X_test = pd.DataFrame(X[:split_point])
Y_train = pd.Series(y[split_point:])
Y_test = pd.Series(y[:split_point])


# 2a
for criteria in ["information_gain", "gini_index"]:
    tree = DecisionTree(criterion=criteria)
    tree.fit(X_train, Y_train)
    y_predicted = tree.predict(X_test)
    print("Criteria :", criteria)
    print("Accuracy: ", accuracy(y_predicted, Y_test))
    for cls in Y_test.unique():
        print("Precision: ", precision(y_predicted, Y_test, cls))
        print("Recall: ", recall(y_predicted, Y_test, cls))


# 2b
cv_outer = KFold(n_splits=5, shuffle=True, random_state=1)

depth = 1
acc_dep = [0.0] * 20
while(depth < 20):
    for train_index, test_index in cv_outer.split(X_pd, y_pd):
        X_train, X_test = X_pd.iloc[train_index].copy(), X_pd.iloc[test_index].copy()
        y_train, y_test = y_pd.iloc[train_index], y_pd.iloc[test_index]
        # print(X_train)
        tree = DecisionTree(max_depth=depth, criterion="information_gain")
        tree.fit(X_train, y_train)
        y_predicted = tree.predict(X_test)
        y_predicted.index = y_test.index

        curr_acc = accuracy(y_predicted, y_test)
        
        if acc_dep[depth] < curr_acc:
            acc_dep[depth] = curr_acc
            
    depth = depth + 1

print(f"Best accuracy: {max(acc_dep)}, best depth: {acc_dep.index(max(acc_dep))}")