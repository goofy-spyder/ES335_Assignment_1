"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """
    cols = X.select_dtypes(include=['object']).columns.tolist()
    encode = OneHotEncoder(sparse_output=False)
    encoded = encode.fit_transform(X[cols])
    encoded_df = pd.DataFrame(encoded, columns=encode.get_feature_names_out(cols))
    df_encoded = pd.concat([X, encoded_df], axis=1)
    df_encoded = df_encoded.drop(cols, axis=1)
    return df_encoded

def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    y = y.dropna()
    if pd.api.types.is_object_dtype(y) or pd.api.types.is_categorical_dtype(y) or pd.api.types.is_bool_dtype(y):
        return False
    
    if pd.api.types.is_numeric_dtype(y):
        unique_values = y.nunique()
        if unique_values<=10:
            return False
        else:
            return True
    return True   

def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    labels = Y.unique()
    entropy = 0
    for label in labels:
        p_cls = len(Y[Y==label])/len(Y)
        entropy+= -p_cls*np.log2(p_cls)
    return entropy

def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    labels = Y.unique()
    gini = 0
    for cls in labels:
        probability_cls = len(Y[Y==cls]) / len(Y)
        gini+=probability_cls**2
    return 1-gini

def mean_squared_error(y:pd.Series):
    if len(y) == 0:
        return 0
    mean = y.mean()
    return ((y - mean)**2).mean()

def information_gain(Y: pd.Series,Y_low: pd.Series, Y_high:pd.Series, criterion: None) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini index or MSE)
    """
    # print(Y_low)
    gain = 0
    weight_l = len(Y_low) / len(Y)
    weight_h = 1 - weight_l
    if(criterion == "gini_index"):
        gain = gini_index(Y) - (weight_l* gini_index(Y_low) + weight_h*gini_index(Y_high))
    elif (criterion == "information_gain"):
        gain  = (entropy(Y) - (weight_l*entropy(Y_low) + weight_h*entropy(Y_high)))
    else:
        gain  = ((len(Y_low) * mean_squared_error(Y_low)) + (len(Y_high)*mean_squared_error(Y_high))) / len(Y)
    
    return gain



def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion=None):

    best_split = {}
    max_info_gain = -float("inf")
    min_mse = float("inf")
    for cols in X.columns:
        # print(cols)
        col_values = X[cols]
        possible_thresholds = np.unique(col_values)
        for thresholds in possible_thresholds:
            data_split_low, data_split_high = split_data(X, y, cols,thresholds)
            if(len(data_split_low)>0 and len(data_split_high)>0):
                low_op, high_op = data_split_low.iloc[:,-1],data_split_high.iloc[:,-1]
                curr_info_gain = information_gain(y, low_op, high_op, criterion)
                if criterion == "mse":
                    if curr_info_gain < min_mse:
                        best_split["index"] = cols
                        best_split["threshold"] = thresholds
                        best_split["data_left"] = data_split_low
                        best_split["data_right"] = data_split_high
                        best_split["mse"] = curr_info_gain
                        min_mse = curr_info_gain
                else:    
                    if curr_info_gain > max_info_gain:
                        best_split["index"] = cols
                        best_split["threshold"] = thresholds
                        best_split["data_left"] = data_split_low
                        best_split["data_right"] = data_split_high
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain
    return best_split

    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """

    # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).


def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    # print(y)
    X['y'] = y
    dataset_low = pd.DataFrame([row for _, row in X.iterrows() if row[attribute]<=value])
    dataset_high = pd.DataFrame([row for _,row in X.iterrows() if row[attribute]>value])
    return dataset_low, dataset_high
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """

    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.
