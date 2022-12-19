"""Script used to help select features for modelling."""

import pandas as pd
import statsmodels.api as sm
from sklearn.tree import DecisionTreeRegressor


def filter_method(input_df, target_variable):
    """Function to apply filter method for feature selection"""
    input_df = input_df.copy()
    input_df["target"] = target_variable  # add back our target column
    # looking at correlation numbers between team1_fouls with the other attributes
    input_df_corr = input_df.corr()["target"].sort_values(ascending=False)
    # taking ten most related variables and taking the index and creating a list
    most_correlated_features = input_df_corr.apply(lambda x: (x * x) ** 0.5).sort_values(ascending=False)[1:11].index
    return most_correlated_features


def forward_selection(input_df, target_variable, significance_level=0.05):
    """Function to apply forward selection method for feature selection"""
    initial_features = list(input_df.columns)
    best_features = []
    while len(initial_features) > 0:
        remaining_features = list(set(initial_features) - set(best_features))
        new_pval = pd.Series(index=remaining_features, dtype="float64")
        for feature in remaining_features:
            model = sm.OLS(target_variable, sm.add_constant(input_df[best_features + [feature]])).fit()
            new_pval[feature] = model.pvalues[feature]
        min_p_value = new_pval.min()
        if min_p_value < significance_level:
            best_features.append(new_pval.idxmin())
        else:
            break
    return best_features


def backward_elimination(input_df, target_variable, significance_level=0.05):
    """Function to apply backward selection method for feature selection"""
    features = list(input_df.columns)
    while len(features) > 0:
        features_with_constant = sm.add_constant(input_df[features])
        p_values = sm.OLS(target_variable, features_with_constant).fit().pvalues[1:]
        max_p_value = p_values.max()
        if max_p_value >= significance_level:
            excluded_feature = p_values.idxmax()
            features.remove(excluded_feature)
        else:
            break
    return features


def feature_importance(input_df, target_variable):
    """Function to apply backward selection method for feature selection"""
    X = input_df
    y = target_variable

    # define the model
    model = DecisionTreeRegressor()
    model.fit(X, y)  # fit the model
    importance_values = model.feature_importances_  # get importance

    # get feature importance for variables
    feature_importance_tuple = []

    for feature_index, importance in enumerate(importance_values):
        feature_importance_tuple.append((feature_index, importance))

    # sort feature importance
    feature_importance_tuple.sort(key=lambda x: x[1], reverse=True)

    # filter for significant features
    filter_features = [feature for feature in feature_importance_tuple if feature[1] > 0.05]
    selected_features_indexes = []
    for feature in filter_features:
        selected_features_indexes.append(feature[0])
    selected_features = input_df.columns[selected_features_indexes].to_list()
    return selected_features


def get_feature_selection_variables_dict(input_df, target_variable):
    """Function used to output a dictionary with keys as feature selection methods and values the variables."""
    feature_selection_variables_dict = {}
    feature_selection_variables_dict["filter_method"] = filter_method(input_df, target_variable)
    feature_selection_variables_dict["forward_selection"] = forward_selection(input_df, target_variable)
    feature_selection_variables_dict["backward_elimination"] = backward_elimination(input_df, target_variable)
    feature_selection_variables_dict["feature_importance"] = feature_importance(input_df, target_variable)
    return feature_selection_variables_dict
