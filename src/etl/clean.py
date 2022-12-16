"""Script used to help clean fouls data set."""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from src.etl.encoding import mean_hot_encoding, one_hot_encoding


def replace_nulls_for_numerical_columns(numerical_cols_w_missing_values, input_df):
    """Function used to replace null values in numerical columns"""
    # replacing our null values
    simple_imputer = SimpleImputer(strategy="mean")
    imputed_values = simple_imputer.fit_transform(input_df[numerical_cols_w_missing_values])
    imputed_values_df = pd.DataFrame(imputed_values, columns=numerical_cols_w_missing_values)

    # drop old columns with missing data (numeric), named column
    input_df = input_df.drop(numerical_cols_w_missing_values, axis=1)
    # resetting index here so it matches with the index with numeric_missing
    input_df.reset_index(drop=True, inplace=True)

    # add fouls with numeric_missing (ie adding two dataframes together side by side)
    imputed_df = pd.concat([input_df, imputed_values_df], axis=1)
    return imputed_df


def hour_rounding(minutes, hour):
    """Function to calculate rounded hour column"""
    if minutes >= 30:
        return hour + 1
    else:
        return hour


def add_date_time_columns(input_df):
    """Function used to create datetime columns from string datetime column."""
    # add columns for year, month, day
    input_df["year"] = input_df["kick_off_datetime"].apply(lambda datetime: float(datetime[2:4]))
    input_df["month"] = input_df["kick_off_datetime"].apply(lambda datetime: float(datetime[5:7]))
    input_df["day"] = input_df["kick_off_datetime"].apply(lambda datetime: float(datetime[8:10]))
    # add column for minutes, hour, rounded_hour
    input_df["time"] = input_df["kick_off_datetime"].apply(lambda datetime: datetime[-8:])
    input_df["minutes"] = input_df["time"].apply(
        lambda datetime: float(datetime[3:5])
    )  # make a minutes columna and change it to a numeric output
    input_df["hour"] = input_df["time"].apply(lambda date: float(date[0:2]))
    input_df["rounded_hour"] = input_df.apply(lambda time: hour_rounding(time["minutes"], time["hour"]), axis=1)
    # drop extra columns
    input_df.drop(columns=["kick_off_datetime", "time"], inplace=True)
    return input_df


def clean_fouls_dataset(fouls_df):
    """Function used to clean fouls data"""

    # drop data from italian second division (has lots of missing data)
    fouls_df = fouls_df.loc[lambda dfr_: dfr_.competition_name != "ItaSB"]

    # replacing our null values
    numerical_columns = fouls_df.columns[fouls_df.dtypes != object]
    columns_with_missing_values = fouls_df.columns[fouls_df.isna().any()]
    numerical_cols_w_missing_values = columns_with_missing_values.intersection(numerical_columns)
    if len(numerical_cols_w_missing_values) > 0:
        fouls_df = replace_nulls_for_numerical_columns(numerical_cols_w_missing_values, fouls_df)

    # replace missing referee data
    most_used_ref_dict = {}

    football_competition_seasons = set(zip(fouls_df["season"], fouls_df["competition_name"]))

    for season, competition in football_competition_seasons:
        most_common_ref = (
            fouls_df.loc[(fouls_df["competition_name"] == competition) & (fouls_df["season"] == season), "referee"]
            .value_counts()
            .index[0]
        )
        most_used_ref_dict[competition + "_" + season] = most_common_ref

    fouls_df["referee"] = fouls_df.apply(
        lambda x: most_used_ref_dict[x.competition_name + "_" + x.season] if x.referee is np.nan else x.referee, axis=1
    )

    return fouls_df


def categorical_variables_to_numeric(cleaned_fouls_df, target_variable="team1_fouls"):
    """Function used to change all non numerical columns in the fouls data to numeric"""
    # change all variables (that are not already) into numeric
    fouls_df = cleaned_fouls_df

    # datetime column to numerical columns
    fouls_df = add_date_time_columns(fouls_df)

    # seasons
    fouls_df["season"] = fouls_df["season"].apply(lambda season: season[2:4])

    # one-hot encoding
    one_hot_encoding_columns_list = ["country", "competition_name"]
    for column in one_hot_encoding_columns_list:
        fouls_df = one_hot_encoding(encoding_variable=column, input_df=fouls_df)

    # mean-hot encoding
    mean_hot_encoding_columns_list = ["team1_name", "team2_name", "referee"]
    for column in mean_hot_encoding_columns_list:
        fouls_df = mean_hot_encoding(encoding_variable=column, target_variable=target_variable, input_df=fouls_df)

    # drop columns
    columns_to_drop = ["competition_category", "season", "team1_name", "team2_name", "referee", "fixture_id"]
    fouls_df.drop(columns=columns_to_drop, inplace=True)

    return fouls_df
