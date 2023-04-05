# Author: Ty Andrews
# Date: 2023-04-04
import os
import sys

import logging
import pandas as pd
import requests
import datetime as dt
from alive_progress import alive_bar

cur_dir = os.getcwd()
SRC_PATH = cur_dir[
    : cur_dir.index("fortunato-wheels-engine") + len("fortunato-wheels-engine")
]
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

# used for getting forex exchange rates and not critical to verify SSL
requests.packages.urllib3.disable_warnings()

# Create a custom logger
logger = logging.getLogger(__name__)


def preprocess_raw_cargurus_data(nrows: int = None, export: bool = True):
    """Preprocesses the raw cargurus data.

    Parameters
    ----------
    nrows : int, optional
        The number of rows to read in, by default None which reads in all rows.
    export : bool, optional
        Whether to export the preprocessed data to a parquet, by default True

    Returns
    -------
    pd.DataFrame
        The preprocessed dataframe

    Raises
    ------
    FileNotFoundError
        If the raw cargurus data is not found

    Notes
    -----
    The following columns are renamed to match the kijiji data:
        - price -> price_usd
        - make_name -> make
        - model_name -> model
    """
    logger.info("Loading in raw cargurus data...")

    raw_df = pd.read_csv(
        os.path.join(SRC_PATH, "data", "raw", "cargurus-vehicle-ads.csv"),
        parse_dates=["listed_date"],
        nrows=nrows,
        low_memory=False,
    )

    raw_df = raw_df.rename(
        columns={
            # old name : new name
            "price": "price_usd",
            "make_name": "make",
            "model_name": "model",
        },
    )

    categorical_cols = [
        "make",
        "model",
        "fuel_type",
        "wheel_system",
    ]

    int_cols = [
        "year",
        "price_usd",
        "horsepower",
    ]

    for cat_col in categorical_cols:
        raw_df[cat_col] = raw_df[cat_col].astype("category")

    for int_col in int_cols:
        raw_df[int_col].fillna(-10, inplace=True)
        raw_df[int_col] = raw_df[int_col].astype("int32")

    raw_df["age_at_posting"] = (raw_df.listed_date.dt.year - raw_df.year).astype("int8")
    raw_df["mileage_per_year"] = (raw_df.mileage / raw_df.age_at_posting).round(0)

    with alive_bar(len(raw_df.listed_date.dt.date.unique()), force_tty=True) as bar:
        for date in raw_df.listed_date.dt.date.unique():
            raw_df.loc[
                raw_df.listed_date.dt.date == date, "exchange_rate_usd_to_cad"
            ] = get_usd_to_cad_exchange(date)
            bar()

    raw_df["price"] = (
        (raw_df.price_usd * raw_df.exchange_rate_usd_to_cad).round(0).astype("int32")
    )

    raw_df["currency"] = "CAD"

    if export:
        save_preprocessed_data(raw_df)
    else:
        return raw_df


def save_preprocessed_data(df):
    """Saves the preprocessed data to a parquet file.

    Parameters
    ----------
    df : pd.DataFrame
        The preprocessed dataframe

    Raises
    ------
    FileNotFoundError
        If the processed data directory is not found
    """
    df.to_parquet(
        os.path.join(SRC_PATH, "data", "processed", "processed-cargurus-ads.parquet"),
    )


def get_usd_to_cad_exchange(date):
    """Gets the USD to CAD exchange rate for a given date.

    Parameters
    ----------
    date : dt.date
        The date to get the exchange rate for

    Returns
    -------
    float
        The exchange rate from USD to CAD

    Raises
    ------
    TypeError
        If the date is not a datetime.date object

    Notes
    -----
    The API used is https://theforexapi.com/
    """
    # check date is valid date object
    if not isinstance(date, dt.date):
        raise TypeError("date must be a datetime.date object")

    date_string = date.strftime("%Y-%m-%d")
    resp = requests.get(
        f"https://theforexapi.com/api/{date_string}/?symbols=CAD&base=USD", verify=False
    ).json()

    return resp["rates"]["CAD"]


if __name__ == "__main__":
    # process the raw cargurus data
    preprocess_raw_cargurus_data()
