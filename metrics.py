from typing import Union
import pandas as pd


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert len(y_hat) == len(y)
    correct_predictions = (y_hat == y).sum()
    total_predictions = len(y)
    return correct_predictions / total_predictions

def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    tp = 0
    fp = 0
    assert len(y_hat) == len(y)
    for y,y_ in zip(y,y_hat):
        if y == cls and y_ == cls:
            tp = tp + 1
        if y != cls and y_ == cls:
            fp = fp + 1
    return tp/(tp+fp)


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    assert len(y_hat) == len(y)
    tp = 0
    fn= 0
    assert len(y_hat) == len(y)
    for y,y_ in zip(y,y_hat):
        if y == cls and y_ == cls:
            tp = tp + 1
        if y == cls and y_ != cls:
            fn = fn + 1
    return tp/(tp+fn)


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    assert len(y_hat) == len(y)
    res = sum((a-b)**2 for a,b in zip(y,y_hat))
    res = res/len(y)
    return res**0.5


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    assert len(y_hat) == len(y)
    res = sum(abs(a-b) for a,b in zip(y,y_hat))
    res = res/len(y)
    return res