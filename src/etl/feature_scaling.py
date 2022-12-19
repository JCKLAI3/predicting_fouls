"""Script used to scale features used for prediction."""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def standardize_data(input_df):
    """Function used to standardize data"""
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(input_df)
    standard_df = pd.DataFrame(standardized_data, columns=input_df.columns)
    return standard_df


def normalize_data(input_df):
    """Function used to normalize data"""
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(input_df)
    normalized_df = pd.DataFrame(normalized_data, columns=input_df.columns)
    return normalized_df
