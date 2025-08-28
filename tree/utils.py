"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
import numpy as np

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """
    return pd.get_dummies(X)

def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    if y.dtype in ["category", "object", "string"]:
        return False
    # if y.nunique() / len(y) <= 0.2:
    #     return False
    return True

def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    class_count = Y.value_counts()
    num_data = Y.shape[0]
    entropy_ = 0.0
    for cls in class_count:
        if cls == 0:
            continue
        p = cls/num_data
        entropy_ += -p*np.log2(p)
    return entropy_


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    class_count = Y.value_counts()
    # print(class_count)
    num_data = Y.shape[0]
    gini_index_ = 0.0
    for cls in class_count:
        if cls == 0:
            continue
        p = cls/num_data
        gini_index_ += p**2
    return 1-gini_index_


def mean_squared_error(Y: pd.Series) -> float:
    """
    Function to calculate the mean squared error
    """
    return np.mean((Y - np.mean(Y))**2)


def information_gain(Y: pd.Series, attr: pd.Series, criterion: str = None, val: float = None) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini index or MSE)
    """
    before_split = 0.0
    after_split = 0.0    
    if check_ifreal(Y):
        before_split += mean_squared_error(Y)
    else:
        if criterion == "entropy":
            before_split += entropy(Y)
        else:
            before_split += gini_index(Y)
    if not check_ifreal(attr): #for discrete input.

        temp = Y.groupby(attr)

        for _, subset in temp:
            weight = len(subset)/len(Y)
            if weight == 0:
                continue
            if check_ifreal(Y):
                after_split += mean_squared_error(subset) * weight
            else:
                if criterion == "entropy":
                    after_split += entropy(subset) * weight
                else:
                    after_split += gini_index(subset) * weight

    else: # for continuos input
        left = Y[attr < val]
        right = Y[attr >= val]


        if check_ifreal(Y):
            after_split += mean_squared_error(left) * len(left)/len(Y)
            after_split += mean_squared_error(right) * len(right)/len(Y)
        else:
            if criterion == "entropy":
                after_split += entropy(left) * len(left)/len(Y)
                after_split += entropy(right) * len(right)/len(Y)
            else:
                after_split += gini_index(left) * len(left)/len(Y)
                after_split += gini_index(right) * len(right)/len(Y)

    return before_split - after_split
            


def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """
    max_gain = -1*1e9
    best_attr = None
    val_split = None
    # print(features)
    for idx, ftr in enumerate(features):
        if check_ifreal(X[ftr]):
            temp = pd.concat([X[ftr].rename("attr"), y.rename("y")], axis=1)
            temp = temp.sort_values(by = "attr", ascending = True).reset_index(drop = True)
            splits = []
            for i in range(len(temp)-1):
                splits.append((temp.loc[i,"attr"] + temp.loc[i+1,"attr"])/2) if temp.loc[i,"y"] != temp.loc[i+1,"y"] else None
            splits = np.array(splits)
            for split in splits:

                gain = information_gain(y, X[ftr],val = split)
                if gain > max_gain:
                    max_gain = gain
                    best_attr = ftr
                    val_split = split
        else:
            gain = information_gain(y, X[ftr], criterion = criterion)
            if gain > max_gain:
                max_gain = gain
                best_attr = ftr
                val_split = None
    # print(best_attr, max_gain, val_split)
    return best_attr, max_gain, val_split
    

    # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).



def split_data(X: pd.DataFrame, y: pd.Series, attribute, val_split = None):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """
    features = X.columns
    # print(features)
    # attribute = features[attribute]
    if val_split is not None:
        return{ "left": ( (X[X[attribute] < val_split]).reset_index(drop = True), (y[X[attribute] < val_split]).reset_index(drop = True)), 
        "right": ((X[X[attribute] >= val_split]).reset_index(drop = True), (y[X[attribute] >= val_split]).reset_index(drop = True))}
    
    else:
        splits = {}
        for (x_val, X_sub), (_, y_sub) in zip (X.groupby( by = attribute), y.groupby(by = X[attribute])):
            X_sub = X_sub.drop(columns=[attribute], axis = 1)
            # if(len(x_val) == 1):
            #     print(x_val)
            #     continue
            splits[x_val] = (X_sub.reset_index(drop = True), y_sub.reset_index(drop = True))
        return splits

    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.
