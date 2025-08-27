from typing import Union
from typing_extensions import Type
import pandas as pd
import numpy as np


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    # assert y_hat.size == y.size
    # TODO: Write here
    # y_hat = y_hat.astype("category")
    # print(y_hat.dtype, y.dtype)
    accuracy = (y.astype(int).reset_index(drop=True) == y_hat.reset_index(drop=True)).mean()
    return accuracy


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    y = y.astype(int).reset_index(drop=True)
    y_hat = y_hat.astype(int).reset_index(drop=True)
    tp = sum((y == cls) & (y_hat == cls))  
    fp = sum((y != cls) & (y_hat == cls))   

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return precision
    # pass


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    y = y.astype(int).reset_index(drop=True)
    y_hat = y_hat.astype(int).reset_index(drop=True)
    tp = sum((y == cls) & (y_hat == cls))  
    fp = sum((y != cls) & (y_hat == cls))   
    total_positives = sum(y==cls)
    recall = tp / total_positives if (total_positives) > 0 else 0.0
    return recall


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    # if y_hat is None:
    #     return 1
    return np.sqrt(np.mean((y_hat - y)**2))


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    # if y_hat is None:
    #     return 1
    # print(y_hat)
    return np.mean(np.abs(y_hat - y))
