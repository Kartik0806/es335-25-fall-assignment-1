import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
import time

def time_taken(N = 30, P = 5):
    tasks = ["RR", "RD", "DD", "DR"]
    time_taken_for_fit = {x: [] for x in tasks}
    time_taken_for_predict = {x: [] for x in tasks}
    num_average_time = 10
    for i in range(num_average_time):
        for task in tasks:
            if task == "RR":
                X = pd.DataFrame({i: pd.Series(np.random.randn(N), dtype="category") for i in range(5)})
                y = pd.Series(np.random.randn(N), dtype="category")
                tree = DecisionTree(criterion="information_gain")  # Split based on Inf. Gain
                start = time.time()
                tree.fit(X, y)
                end = time.time()
                time_taken_for_fit[task].append(end - start)
                start = time.time()
                y_hat = tree.predict(X)
                end = time.time()
                time_taken_for_predict[task].append(end - start)
            elif task == "RD":
                X = pd.DataFrame({i: pd.Series(np.random.randn(N), dtype="category") for i in range(5)})
                y = pd.Series(np.random.randint(P, size=N), dtype="category")
                tree = DecisionTree(criterion="information_gain")  # Split based on Inf. Gain
                start = time.time()
                tree.fit(X, y)
                end = time.time()
                time_taken_for_fit[task].append(end - start)
                start = time.time()
                y_hat = tree.predict(X)
                end = time.time()
                time_taken_for_predict[task].append(end - start)
            elif task == "DD":

                X = pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(5)})
                y = pd.Series(np.random.randint(P, size=N), dtype="category")
                tree = DecisionTree(criterion="information_gain")  # Split based on Inf. Gain
                start = time.time()
                tree.fit(X, y)
                end = time.time()
                time_taken_for_fit[task].append(end - start)
                start = time.time()
                y_hat = tree.predict(X)
                end = time.time()
                time_taken_for_predict[task].append(end - start)
            elif task == "DR":
                X = pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(5)})
                y = pd.Series(np.random.randn(N), dtype="category")
                tree = DecisionTree(criterion="information_gain")  # Split based on Inf. Gain
                start = time.time()
                tree.fit(X, y)
                end = time.time()
                time_taken_for_fit[task].append(end - start)
                start = time.time()
                y_hat = tree.predict(X)
                end = time.time()
                time_taken_for_predict[task].append(end - start)

    # Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
    avg_fit_time = {}
    avg_predict_time = {}
    for task in tasks:
        avg_fit_time[task] = np.mean(time_taken_for_fit[task])
        avg_predict_time[task] = np.mean(time_taken_for_predict[task])
        print(task)
        print("Average time taken for fit: ", np.mean(time_taken_for_fit[task]))
        print("Average time taken for predict: ", np.mean(time_taken_for_predict[task]))
        print("Standard deviation for fit: ", np.std(time_taken_for_fit[task]))
        print("Standard deviation for predict: ", np.std(time_taken_for_predict[task]))
    

    return avg_fit_time, avg_predict_time


def plot_time(N, P):
    avg_fit, avg_predict = time_taken(N, P)

    tasks = list(avg_fit.keys())

    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.bar(tasks, avg_fit.values())
    plt.title(f"Fit Time (N={N}, P={P})")
    plt.ylabel("Seconds")

    plt.subplot(1,2,2)
    plt.bar(tasks, avg_predict.values())
    plt.title(f"Predict Time (N={N}, P={P})")
    plt.ylabel("Seconds")

    plt.show()


interact(
    plot_time, 
    N=IntSlider(min=10, max=500, step=10, value=30, description="N"),
    P=IntSlider(min=2, max=50, step=1, value=5, description="P")
)