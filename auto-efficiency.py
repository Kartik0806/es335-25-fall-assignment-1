import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)

# Reading the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, delim_whitespace=True, header=None,
                 names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model year", "origin", "car name"])

# Clean the above data by removing redundant columns and rows with junk values
# Compare the performance of your model with the decision tree module from scikit learn

data = data[~(data == '?').any(axis=1)].reset_index(drop = True)

X = data.drop(columns = ["mpg","car name"])
y = data["mpg"]

X['horsepower'] = data["horsepower"].astype("float")

criteria = "entropy"
tree = DecisionTree(criterion=criteria)  # Split based on Inf. Gain
train_data_size = int(0.7*(len(X)))

X_train, y_train, X_test, y_test = X[:train_data_size], y[:train_data_size], X[train_data_size:], y[train_data_size:]
tree.fit(X_train, y_train)
y_hat = tree.predict(X_test)
# tree.plot()
print(y_hat.shape, y_test.shape)
print("RMSE: ", rmse(y_hat, y))
print("MAE: ", mae(y_hat, y))

print(50*"-", "sklearn", 50*"-")

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

model = DecisionTreeRegressor(criterion="squared_error", max_depth=5)
model.fit(X_train, y_train)
y_hat = model.predict(X_test)
print("RMSE: ", np.sqrt(mean_squared_error(y_test, y_hat)))
print("MAE: ", mean_absolute_error(y_test, y_hat))

