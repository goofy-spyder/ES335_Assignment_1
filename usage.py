"""
This script tests the DecisionTree class on four different
data scenarios:
- Real Input, Real Output (Regression)
- Real Input, Discrete Output (Classification)
- Discrete Input, Discrete Output (Classification)
- Discrete Input, Real Output (Regression)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)

print("Real input Real output\n")
N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))
tree = DecisionTree(criterion="mse")
tree.fit(X, y)
y_hat = tree.predict(X)
tree.plot()
print(f"RMSE: {rmse(y_hat, y):.4f}")
print(f"MAE: {mae(y_hat, y):.4f}")

print("Real input, Discrete output\n")
N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randint(P, size=N), dtype="category")

for criteria in ["information_gain", "gini_index"]:
    tree = DecisionTree(criterion=criteria)
    tree.fit(X, y)
    y_hat = tree.predict(X)
    tree.plot()
    print(f"Criteria: {criteria}")
    print(f"Accuracy: {accuracy(y_hat, y):.4f}")
    for cls in y.unique():
        print(f"  Precision for class {cls}: {precision(y_hat, y, cls):.4f}")
        print(f"  Recall for class {cls}: {recall(y_hat, y, cls):.4f}")

print("Discrete input, Discrete output\n")
N = 30
P = 5
X = pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(5)})
y = pd.Series(np.random.randint(P, size=N), dtype="category")

for criteria in ["information_gain", "gini_index"]:
    tree = DecisionTree(criterion=criteria)
    tree.fit(X, y)
    y_hat = tree.predict(X)
    tree.plot()
    print(f"Criteria: {criteria}")
    print(f"Accuracy: {accuracy(y_hat, y):.4f}")
    for cls in y.unique():
        print(f"  Precision for class {cls}: {precision(y_hat, y, cls):.4f}")
        print(f"  Recall for class {cls}: {recall(y_hat, y, cls):.4f}")
print("Discrete input, real output")
N = 30
P = 5
X = pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(5)})
y = pd.Series(np.random.randn(N))

tree = DecisionTree(criterion="mse")
tree.fit(X, y)
y_hat = tree.predict(X)
tree.plot()
print(f"RMSE: {rmse(y_hat, y):.4f}")
print(f"MAE: {mae(y_hat, y):.4f}")