"""Script used to evaluate model performance."""

import math

from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_mean_squared_error(y_test, y_pred):
    """Function used to calculate mean squared error"""
    return mean_squared_error(y_test, y_pred)


def calculate_root_mean_squared_error(y_test, y_pred):
    """Function used to calculate root mean squared error"""
    return math.sqrt(mean_squared_error(y_test, y_pred))


def calculate_mean_absolute_error(y_test, y_pred):
    """Function used to calculate mean absolute error"""
    return mean_absolute_error(y_test, y_pred)
