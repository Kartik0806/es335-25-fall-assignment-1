import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification

np.random.seed(42)

# Code given in the question
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
# plt.scatter(X[:, 0], X[:, 1], c=y)

# Write the code for Q2 a) and b) below. Show your results.

X = pd.DataFrame(X)
y = pd.Series(y, dtype="category")
criteria = "gini_index"
tree = DecisionTree(criterion=criteria)  # Split based on Inf. Gain
train_data_size = int(0.7*(len(X)))

X_train, y_train, X_test, y_test = X[:train_data_size], y[:train_data_size], X[train_data_size:], y[train_data_size:]
tree.fit(X_train, y_train)
y_hat = tree.predict(X_test)
tree.plot()
print("Criteria :", criteria)
print(y_hat.shape, y_test.shape)
print("Accuracy: ", accuracy(y_hat, y_test))
for cls in y.unique():

    print("Precision: ", precision(y_hat, y_test, cls))
    print("Recall: ", recall(y_hat, y_test, cls))
