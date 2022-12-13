"""Script used to help clean fouls data set."""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


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
