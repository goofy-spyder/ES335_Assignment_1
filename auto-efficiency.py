import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import rmse, mae
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, sep='\s+', header=None,
                   names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                          "acceleration", "model year", "origin", "car name"])

data = data.drop(columns=["car name"])
data = data[data['horsepower'] != '?']

data['horsepower'] = pd.to_numeric(data['horsepower'])
X = data.drop(columns=["mpg"])
y = data["mpg"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

custom_tree = DecisionTree(criterion='mse', max_depth=5)
custom_tree.fit(X_train, y_train)

y_pred_custom = custom_tree.predict(X_test)

custom_rmse = rmse(y_test, y_pred_custom)
custom_mae = mae(y_test, y_pred_custom)
print(f"Custom Decision Tree RMSE: {custom_rmse:.4f}")
print(f"Custom Decision Tree MAE: {custom_mae:.4f}")

sklearn_tree = DecisionTreeRegressor(criterion='squared_error', max_depth=5, random_state=42)
sklearn_tree.fit(X_train, y_train)

y_pred_sklearn = sklearn_tree.predict(X_test)
sklearn_rmse = rmse(y_test, y_pred_sklearn)
sklearn_mae = mae(y_test, y_pred_sklearn)
print(f"Scikit-Learn Decision Tree RMSE: {sklearn_rmse:.4f}")
print(f"Scikit-Learn Decision Tree MAE: {sklearn_mae:.4f}")

plt.figure(figsize=(10, 6))
plt.bar(['Custom Tree', 'Scikit-Learn Tree'], [custom_rmse, sklearn_rmse])
plt.title('RMSE Comparison: Custom vs. Scikit-Learn', fontsize=16)
plt.ylabel('RMSE', fontsize=12)
plt.xlabel('Model', fontsize=12)
plt.show()
