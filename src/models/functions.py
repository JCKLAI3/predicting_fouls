"""Script used to store models used to predict the number of fouls in a match"""
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


def SVMRegression(kernel="rbf"):
    """Defeine SVM regressor model"""
    return SVR(kernel=kernel)


def PoissonRegression(max_iter=10000):
    """Defeine Poisson regressor model"""
    return linear_model.PoissonRegressor(max_iter=max_iter)


def DecisionTreeRegression():
    """Defeine Decision Tree regressor model"""
    return DecisionTreeRegressor()


def RandomForestRegression():
    """Defeine Random Forest regressor model"""
    return RandomForestRegressor()


def XGBRegression():
    """Defeine XGBoost regressor model"""
    return XGBRegressor()
