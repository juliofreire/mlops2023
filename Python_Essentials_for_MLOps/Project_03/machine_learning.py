"""
NBA Player Stats Analysis Module

This module contains functions for loading, processing, and analyzing NBA player 
and MVP stats data. It includes functions for data loading, handling missing values, 
splitting data for training and testing, training machine learning models, evaluating 
model predictions, conducting backtests over multiple years, and adding ranking 
information to predictions.

Functions:
- load_data(): Load NBA player and MVP stats data from a CSV file.
- handle_missing_values(stats): Handle missing values in a DataFrame by filling them with zeros.
- train_test_split(stats): Split a DataFrame into training and testing sets based on the 
"Year" column.
- train_and_predict(train, test, model, predictors): Train a machine learning model and use
it to make predictions.
- evaluate_predictions(test, predictions): Evaluate model predictions and add rankings.
- find_ap(combination, actual): Calculate the Average Precision (AP) of model predictions 
against actual data.
- backtest(stats, model, years, predictors): Perform a backtest of a machine learning model's 
performance over multiple years.
- add_ranks(predictions): Add ranking information to a DataFrame based on predictions.

This module uses machine learning models like Ridge and Random Forest to predict MVP shares and 
evaluates their performance over the years.
"""

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


def load_data():
    """
    Load NBA player and MVP stats data from a CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing NBA player and MVP stats.

    This function reads data from a CSV file named "player_mvp_stats.csv" and 
    returns it as a DataFrame.
    The data is assumed to be in a structured format, and the index column is specified 
    as the first column (index_col=0).
    """
    stats = pd.read_csv("player_mvp_stats.csv", index_col=0)
    return stats

def handle_missing_values(stats):
    """
    Handle missing values in a DataFrame by filling them with zeros.

    Args:
        stats (pd.DataFrame): A DataFrame with potentially missing values.

    Returns:
        pd.DataFrame: A DataFrame with missing values replaced by zeros.

    This function takes a DataFrame as input and replaces any missing (NaN) values with zeros,
    ensuring that all columns have valid numeric values for further analysis or modeling.
    """
    stats = stats.fillna(0)
    return stats

def train_test_split(stats):
    """
    Split a DataFrame into training and testing sets based on the "Year" column.

    Args:
        stats (pd.DataFrame): A DataFrame containing data with a "Year" column.

    Returns:
        pd.DataFrame, pd.DataFrame: Two DataFrames representing the training and testing sets.

    This function takes a DataFrame with a "Year" column and splits it into two sets:
    a training set that includes data from all years except 2021 and a testing set that 
    includes data only from the year 2021.
    """
    train = stats[~(stats["Year"] == 2021)]
    test = stats[stats["Year"] == 2021]
    return train, test

def train_and_predict(train, test, model, predictors):
    """
    Train a machine learning model and use it to make predictions.

    Args:
        train (pd.DataFrame): The training data DataFrame.
        test (pd.DataFrame): The testing data DataFrame.
        model: The machine learning model to train and use for predictions.
        predictors (list): A list of predictor columns to use for training and predictions.

    Returns:
        pd.Series: Predictions made by the model.

    This function takes training and testing data, a machine learning model, and 
    a list of predictor columns.
    It trains the model using the training data and the specified predictors and 
    then uses the trained model to make predictions on the testing data. The function 
    returns the predictions as a Series.
    """
    model.fit(train[predictors], train["Share"])
    predictions = model.predict(test[predictors])
    return predictions

def evaluate_predictions(test, predictions):
    """
    Evaluate model predictions and add rankings.

    Args:
        test (pd.DataFrame): The testing data DataFrame.
        predictions (pd.Series): Predictions made by a machine learning model.

    Returns:
        pd.DataFrame: A DataFrame with model predictions and added rankings.

    This function takes the testing data and predictions made by a machine learning model.
    It creates a DataFrame that includes the predictions and adds ranking information to the 
    DataFrame based on the "predictions" column.
    The resulting DataFrame contains both predictions and rankings.
    """
    combination = pd.DataFrame(predictions, columns=["predictions"], index=test.index)
    combination = add_ranks(combination)
    return combination

def find_ap(combination):
    """
    Calculate Average Precision (AP) for the given combination.

    Args:
        combination (pd.DataFrame): A DataFrame with ranking information.

    Returns:
        float: Average Precision (AP).
    """
    actual = combination.sort_values("Share", ascending=False).head(5)
    predicted = combination.sort_values("predictions", ascending=False)
    ps = []
    found = 0
    seen = 1
    for _, row in predicted.iterrows():
        if row["Player"] in actual["Player"].values:
            found += 1
            ps.append(found / seen)
        seen += 1
    return sum(ps) / len(ps)

def backtest(model, stats, years, predictors):
    """
    Perform backtesting of a model using different years.

    Args:
        model: The machine learning model.
        stats (pd.DataFrame): The DataFrame containing the data.
        years (list): List of years to test on.
        predictors (list): List of predictor variables.

    Returns:
        float: Mean Average Precision (MAP) over all years.
        list: List of Average Precisions (AP) for each year.
        pd.DataFrame: Combined predictions for all years.
    """
    Ap = []
    all_prediction = []
    for year in years:
        train = stats[stats["Year"] < year].copy()
        test = stats[stats["Year"] == year].copy()
        Standardscaler = StandardScaler()
        Standardscaler.fit(train[predictors])
        train[predictors] = Standardscaler.transform(train[predictors])
        test[predictors] = Standardscaler.transform(test[predictors])
        model.fit(train[predictors], train["Share"])
        predictions = train_and_predict(train, test, model, predictors)
        combination = evaluate_predictions(test, predictions)
        all_prediction.append(combination)
        Ap.append(find_ap(combination))
    mean_ap = sum(Ap) / len(Ap)
    return mean_ap, Ap, pd.concat(all_prediction)

def add_ranks(predictions):
    """
    Add ranking information to a DataFrame based on predictions.

    Args:
        predictions (pd.DataFrame): A DataFrame with prediction values.

    Returns:
        pd.DataFrame: A DataFrame with added ranking information.

    This function takes a DataFrame with prediction values and adds ranking information 
    based on the predictions. It computes both the predicted rankings and actual rankings 
    (Rk) and calculates the difference (Diff) between them. The resulting DataFrame includes 
    the added ranking information.
    """
    predictions = predictions.sort_values("predictions", ascending=False)
    predictions["Predicted_Rk"] = list(range(1, predictions.shape[0] + 1))
    predictions = predictions.sort_values("Share", ascending=False)
    predictions["Rk"] = list(range(1, predictions.shape[0] + 1))
    predictions["Diff"] = predictions["Rk"] - predictions["Predicted_Rk"]
    return predictions

stat = load_data()
stat = handle_missing_values(stat)

reg = Ridge(alpha=0.1)
rf = RandomForestRegressor(n_estimators=50, random_state=1, min_samples_split=5)

all_years = list(range(1991, 2022))
predictorsz = ["Age", "G", "GS", "MP", "FG", "FGA", 'FG%', '3P', '3PA', '3P%',
            '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 
            'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'W', 'L', 'W/L%', 'GB', 'PS/G', 
            'PA/G', 'SRS', 'NPos', 'NTm']

mean_aps, aps, all_predictions = backtest(stat, reg, all_years[28:], predictorsz)
print("Ridge Mean AP:", mean_aps)

mean_aps, aps, all_predictions = backtest(stat, rf, all_years[28:], predictorsz)
print("Random Forest Mean AP:", mean_aps)
