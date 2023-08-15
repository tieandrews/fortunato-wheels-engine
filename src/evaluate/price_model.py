# Author: Ty Andrews
# Date: 2023-08-08

import os, sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

SRC_PATH = os.path.join(os.path.dirname(__file__), "..", "..")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from src.logs import get_logger


logger = get_logger(__name__)


def calculate_evaluation_metrics(y, y_pred, metrics=["rmse", "mape", "r2"]):
    """Calculate the evaluation metrics for a regression model.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        The true values of the target.
    y_pred : array-like of shape (n_samples,)
        The predicted values of the target.
    metrics : list of str, optional (default=["rmse", "mape", "r2"])
        The metrics to calculate. Possible values are "rmse", "mape", and "r2".

    Returns
    -------
    metrics_df : pandas.DataFrame
        A dataframe containing the evaluation metrics with columns "metric" and "value".
    """

    # ensure metrics are valid
    valid_metrics = ["rmse", "mape", "r2"]
    if not all([metric in valid_metrics for metric in metrics]):
        raise ValueError(f"Metrics must be a subset of {valid_metrics}")

    if len(metrics) == 0:
        raise ValueError("Must provide at least one metric")

    # Create a dictionary to store the evaluation metrics
    evaluation_metrics = {}

    # Calculate the RMSE if it is requested
    if "rmse" in metrics:
        evaluation_metrics["rmse"] = round(
            mean_squared_error(y, y_pred, squared=False), 1
        )

    # Calculate the MAPE if it is requested
    if "mape" in metrics:
        evaluation_metrics["mape"] = round((abs(y - y_pred) / y).mean(), 4)

    # Calculate the R2 if it is requested
    if "r2" in metrics:
        evaluation_metrics["r2"] = round(r2_score(y, y_pred), 4)

    # Convert the dictionary to a dataframe
    metrics_df = (
        pd.DataFrame.from_dict(evaluation_metrics, orient="index")
        .reset_index()
        .rename(columns={"index": "metric", 0: "value"})
    )

    return metrics_df


def calculate_evaluation_metrics_by_model(df, metrics=["rmse", "mape", "r2"]):
    """Calculate the evaluation metrics for a regression model broken down
    by make and model.

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe containing the true prices and predicted prices for multiple
        car ads. Must contain columns "make", "model", "price", and "predicted_price".
    metrics : list of str, optional (default=["rmse", "mape", "r2"])
        The metrics to calculate. Possible values are "rmse", "mape", and "r2".

    Returns
    -------
    make_model_results_df : pandas.DataFrame
        A dataframe containing the evaluation metrics broken down by make and model.
    """

    # ensure make, model, price, predicted price ar ein columns
    if not all(
        [col in df.columns for col in ["make", "model", "price", "predicted_price"]]
    ):
        raise ValueError(
            "df must contain make, model, price, and predicted_price columns"
        )

    # get unique make model combinations
    unique_make_models = df[["make", "model"]].drop_duplicates()
    unique_makes = df["make"].unique()

    make_model_results = {}

    # calcualte metrics for each make model combination
    for make, model in unique_make_models.values:
        subset_data = df[(df["make"] == make) & (df["model"] == model)]
        y_actual_subset = subset_data["price"]
        y_predicted_subset = subset_data["predicted_price"]

        make_model_results[(make, model)] = {}

        make_model_results[(make, model)]["count"] = len(y_actual_subset)

        if "rmse" in metrics:
            rmse = mean_squared_error(
                y_actual_subset, y_predicted_subset, squared=False
            )
            make_model_results[(make, model)]["RMSE"] = round(rmse, 1)

        if "mape" in metrics:
            mape = np.mean(
                np.abs((y_actual_subset - y_predicted_subset) / y_actual_subset)
            )
            make_model_results[(make, model)]["MAPE"] = round(mape, 4)

        if "r2" in metrics:
            r2 = r2_score(y_actual_subset, y_predicted_subset)
            make_model_results[(make, model)]["R2"] = round(r2, 4)

    make_model_results_df = (
        pd.DataFrame.from_dict(make_model_results, orient="index")
        .reset_index()
        .rename(columns={"level_0": "make", "level_1": "model"})
    )

    return make_model_results_df


def calculate_evaluation_metrics_by_make(df, metrics=["rmse", "mape", "r2"]):
    """Calculate the evaluation metrics for a regression model broken down
    by make.

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe containing the true prices and predicted prices for multiple
        car ads. Must contain columns "make", "price", and "predicted_price".
    metrics : list of str, optional (default=["rmse", "mape", "r2"])
        The metrics to calculate. Possible values are "rmse", "mape", and "r2".

    Returns
    -------
    make_results_df : pandas.DataFrame
        A dataframe containing the evaluation metrics broken down by make and model.
    """

    # ensure make, price, predicted price ar ein columns
    if not all([col in df.columns for col in ["make", "price", "predicted_price"]]):
        raise ValueError(
            "df must contain make, model, price, and predicted_price columns"
        )

    # get unique makes
    unique_makes = df["make"].unique()

    make_results = {}

    # calcualte metrics for each make model combination
    for make in unique_makes:
        subset_data = df[(df["make"] == make)]
        y_actual_subset = subset_data["price"]
        y_predicted_subset = subset_data["predicted_price"]

        make_results[(make)] = {}

        make_results[(make)]["count"] = len(y_actual_subset)

        if "rmse" in metrics:
            rmse = mean_squared_error(
                y_actual_subset, y_predicted_subset, squared=False
            )
            make_results[(make)]["RMSE"] = round(rmse, 1)

        if "mape" in metrics:
            mape = np.mean(
                np.abs((y_actual_subset - y_predicted_subset) / y_actual_subset)
            )
            make_results[(make)]["MAPE"] = round(mape, 4)

        if "r2" in metrics:
            r2 = r2_score(y_actual_subset, y_predicted_subset)
            make_results[(make)]["R2"] = round(r2, 4)

    make_results_df = (
        pd.DataFrame.from_dict(make_results, orient="index")
        .reset_index()
        .rename(columns={"index": "make"})
    )

    return make_results_df


def calculate_train_data_metrics(df):
    """
    Calculates summary statistics of the training data set.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame containing the training data.

    Returns
    -------
    train_data_metrics_df : pandas.DataFrame
        A DataFrame with columns for make, model, mean_price, mean_mileage, mean_age_at_posting and count.
        The mean_price and mean_mileage columns are rounded to 1 decimal place.

    """
    # ensure make, model, price, mileage, and age are in columns
    if not all(
        [
            col in df.columns
            for col in ["make", "model", "price", "mileage_per_year", "age_at_posting"]
        ]
    ):
        raise ValueError(
            "df must contain make, model, age_at_posting, mileage_per_year, and price columns"
        )

    train_data_metrics_df = (
        df.groupby(["make", "model"])
        .agg({"price": "mean", "mileage_per_year": "mean", "age_at_posting": "mean"})
        .reset_index()
        .rename(
            columns={
                "price": "mean_price",
                "mileage_per_year": "mean_mileage",
                "age_at_posting": "mean_age_at_posting",
            }
        )
        .assign(count=df.groupby(["make", "model"]).size().values)
        .round({"mean_price": 1, "mean_mileage": 1, "mean_age_at_posting": 1})
    )

    return train_data_metrics_df
