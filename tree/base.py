"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from ast import Tuple
from dataclasses import dataclass, field
import string
from typing import Literal, List, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *

np.random.seed(42)


@dataclass
class Node: 
    children: Dict[string, "Node"] = field(default_factory = dict)
    value: Optional[float] = None # for continuous input
    split_upon: Optional[string] = None # for splitting attribute 
    label: Optional[string] = None # for majority class
    mean: Optional[float] = None # for real output
    is_leaf: Optional[bool] = False

@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None
        self.features = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """
        self.root = self.build(X, y, 0)

        print("done building")
        # If you wish your code can have cases for different types of input and output data (discrete, real)
        # Use the functions from utils.py to find the optimal attribute to split upon and then construct the tree accordingly.
        # You may(according to your implemetation) need to call functions recursively to construct the tree. 


    def build(self, X: pd.DataFrame, y: pd.Series, height: int = 0):

        # print(y.nunique())
        if height >= self.max_depth:  # max depth reached
            if check_ifreal(y):
                return Node(mean=y.mean(), children={}, is_leaf=True)
            else:
                return Node(label=y.mode()[0], children={}, is_leaf=True)

        elif y.nunique() == 1:  # pure leaf
            if check_ifreal(y):
                return Node(mean=y.mean(), children={}, is_leaf=True)
            else:
                return Node(label=y.mode()[0], children={}, is_leaf=True)

        # elif y.nunique() == 0:
        #     return None

        else:
            best_attr, max_gain, val_split = opt_split_attribute(X, y, self.criterion, X.columns)

            if max_gain <= 0:
                if check_ifreal(y):
                    return Node(mean=y.mean(), children={}, is_leaf=True)
                else:
                    return Node(label=y.mode()[0], children={}, is_leaf=True)


            split_children = split_data(X, y, attribute=best_attr, val_split=val_split)

            if(check_ifreal(y)):
                node = Node(split_upon = best_attr, value=val_split,
                            children={}, is_leaf=False, mean = y.mean())
            else:
                node = Node(split_upon = best_attr, value=val_split,
                            children={}, label=y.mode()[0], is_leaf=False)

            for key, (X_sub, y_sub) in split_children.items():
                
                if(X_sub.shape[0] == 0):
                    continue
                node.children[key] = self.build(X_sub, y_sub, height + 1)

            return node
    
    def traverse(self, node: Node, X):
        if node.is_leaf:
            # print(node.split_upon, node.value, node.label, node.mean)
            return node.label, node.mean

        else:
            feature = node.split_upon
            # print(X[feature].iloc[0])

            if check_ifreal(X[feature]):
                # print(node.mean)
                if X[feature].iloc[0] < node.value:
                    x =  self.traverse(node.children["left"], X)
                    # print(x)
                    return x
                else:
                    x = self.traverse(node.children["right"], X)
                    # print(x)
                    return x
            else:
                for key, child in node.children.items():
                    # print(key)
                    if(X[feature].iloc[0] == key):
                        x = self.traverse(child, X)
                        # print(x)
                        return x
        # print(node.split_upon, node.value, node.label, node.mean)
        # return node.label, node.mean

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """
        y_hat = []
        for i in range(len(X)):
            y_ = self.traverse(self.root, X[i:i+1])
            # print(y_)
            y_hat.append(y_[0] if y_[0] is not None else y_[1])
        return pd.Series(y_hat)
        # Traverse the tree you constructed to return the predicted values for the given test inputs.

        

    def plot(self) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        pass
