"""Script used for encoding categorical variables to numeric"""

import pandas as pd


def one_hot_encoding(encoding_variable, input_df):
    """Function used to apply one hot encoding"""
    column_dummies = pd.get_dummies(input_df[encoding_variable], drop_first=True)  # one-hot encoding
    input_df = pd.concat([input_df.drop(encoding_variable, axis=1), column_dummies], axis=1)
    return input_df


def mean_hot_encoding(encoding_variable, target_variable, input_df):
    """Function used to apply mean hot encoding."""

    # compute the global mean
    mean = input_df[target_variable].mean()

    # compute the number of values and the mean of each group
    agg = input_df.groupby(encoding_variable)[target_variable].aggregate(["count", "mean"])
    counts = agg["count"]
    means = agg["mean"]
    weight = 100

    # compute the 'smoothed' means
    smooth = (counts * means + weight * mean) / (counts + weight)
    # Replace each value by the according smoothed mean
    input_df[f"{encoding_variable}_enc"] = input_df[encoding_variable].map(smooth)

    return input_df
