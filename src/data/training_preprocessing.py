# Author: Ty Andrews
# Date: 2023-07-28

import os, sys

import pandas as pd

SRC_PATH = os.path.join(os.getcwd(), "..", "..")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from src.logs import get_logger

logger = get_logger(__name__)

def preprocess_ads_for_training(
        ads_df:pd.DataFrame,
        model_features = [
            "age_at_posting",
            "mileage_per_year",
            "make",
            "model",
            "wheel_system",
        ],
        min_num_ads = 1000,
        max_age_at_posting = 20,
        min_price = 1000,
        max_price = 250000,
    ):
    """
    Preprocess a pandas DataFrame containing car ads for training.

    Args:
        ads_df (pd.DataFrame): A pandas DataFrame containing the car ads to preprocess.
        model_features (list): The features of the ads to keep in the preprocessed DataFrame.
        min_num_ads (int): The minimum number of ads needed for a make/model combo to be kept.
        max_age_at_posting (int): The maximum age at posting for an ad to be kept.
        min_price (int): The minimum price for an ad to be kept.
        max_price (int): The maximum price for an ad to be kept.

    Returns:
        preprocessed_df (pd.DataFrame): A pandas DataFrame containing the preprocessed car ads.
    """

    logger.info(f"Preprocessing ads for training, starting with {len(ads_df)} ads")

    if "model" not in model_features:
        model_features.append("model")

    if "price" not in model_features:
        model_features = model_features + ["price"]

    preprocessed_df = ads_df[model_features].copy()

    # remove NaN models and "other"
    preprocessed_df = preprocessed_df[~preprocessed_df["model"].isna()]
    preprocessed_df = preprocessed_df[preprocessed_df["model"].str.lower() != "other"]

    # remove ads with prices outside of min_price and max_price
    preprocessed_df = preprocessed_df.query("price > @min_price & price < @max_price")

    if "age_at_posting" in model_features:
        # remove cars older than max_age_at_posting years
        preprocessed_df = preprocessed_df[preprocessed_df["age_at_posting"] <= max_age_at_posting]

    if "wheel_system" in model_features:
        # replace NaN wheel_system with "unknown"
        preprocessed_df["wheel_system"] = preprocessed_df["wheel_system"].fillna("unknown")

    if "mileage_per_year" in model_features:
        # where ads have an age_at_posting of zero set mileage_per_year to 0
        preprocessed_df.loc[preprocessed_df["age_at_posting"] == 0, "mileage_per_year"] = 0
        # drop any other mileage per year NaNs
        preprocessed_df = preprocessed_df[~preprocessed_df["mileage_per_year"].isna()]

    # ensure only unique make/model combos are kept, some car models share names across makes
    # inspired from: https://stackoverflow.com/questions/35268817/unique-combinations-of-values-in-selected-columns-in-pandas-data-frame-and-count
    make_models_to_keep = (
        preprocessed_df
        .groupby(["make", "model"])
        .size()
        .reset_index()
        .rename(columns={0: "count"})
        .query("count > @min_num_ads")
    )

    # inner join on make and model to remove any makes that have less than min_num_ads
    preprocessed_df = preprocessed_df.merge(make_models_to_keep.drop(columns=["count"]), on=["make", "model"])

    logger.info(f"Preprocessing ads for training, ending with {len(preprocessed_df)} ads")

    return preprocessed_df