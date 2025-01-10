# Author: Ty Andrews
# Date: 2023-04-05
import os
import sys

import logging
import pandas as pd
import numpy as np
import json
from collections import defaultdict

SRC_PATH = os.path.join(os.path.dirname(__file__), "..", "..")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from src.data.cosmos_db import connect_to_database
from src.logs import get_logger

# Create a custom logger
logger = get_logger(__name__)


class CarAds:
    def __init__(self):
        self.year_range = None
        self.make = None
        self.model = None
        self.df = None
        self.sources = []

    def get_car_ads(
        self,
        year_range: tuple = None,
        make: str = None,
        model: str = None,
        sources: list = ["cargurus", "kijiji"],
        data_dump: str = None,
        limit_ads: int = None,
    ):
        """Gets all car ads from all sources and assigns to CarAds.df object.

        Parameters
        ----------
        year_range : tuple, optional
            A tuple containing the min and max year of the car ads to get, by default None
        make : str, optional
            The make of the car ads to get, by default None
        model : str, optional
            The model of the car ads to get, by default None
        sources: list
            Which data sources to load from, options are cargurus, kijiji.
            By default is ["cargurus", "kijiji"].
        data_dump: str
            The path to a parquet file containing car ads to load.
        limit_ads:
            How many ads to load, if None loads all.

        Raises
        ------
        ValueError
            If year_range is not a tuple.

        Notes
        -----
        If year_range is None, then all car ads are returned.
        If make is None, then all car ads for the given year_range are returned.
        If model is None, then all car ads for the given year_range and make are returned.
        """

        self.sources = sources

        if data_dump is not None:
            logger.info(f"Loading car ads from {data_dump}...")

            if data_dump.endswith(".csv"):
                if isinstance(limit_ads, int):
                    logger.info(f"Loading first {limit_ads} ads...")
                    self.df = pd.read_csv(
                        data_dump, parse_dates=["listed_date"], nrows=limit_ads
                    )
                else:
                    self.df = pd.read_csv(data_dump, parse_dates=["listed_date"])
                return
            elif data_dump.endswith(".parquet"):
                self.df = pd.read_parquet(data_dump)
                return
            else:
                raise ValueError(
                    f"Invalid file type for data_dump. Must be .csv or .parquet."
                )

        if year_range is None:
            self.year_range = (1900, 2050)
        else:
            self.year_range = year_range

        self.make = make
        self.model = model

        car_ads_df = pd.DataFrame()

        logger.info(f"Getting all car ads from {sources} sources...")
        if ("kijiji" in sources) | (limit_ads is not None):
            # get kijiji ads
            if limit_ads is not None:
                kijiji_df = self._get_kijiji_ads(limit_ads=limit_ads)
                # if limit_ads is passed, gets only ads from kijiji then returns as parquet reading has issues
                # limiting number of rows read in
                self.df = pd.concat([car_ads_df, kijiji_df], ignore_index=True)
                # update sources to reflect only kijiji ads present
                self.sources = ["kijiji"]
                return
            else:
                kijiji_df = self._get_kijiji_ads(limit_ads=limit_ads)
                car_ads_df = pd.concat([car_ads_df, kijiji_df], ignore_index=True)

        if "cargurus" in sources:
            # get cargurus ads
            cargurus_df = self._get_cargurus_ads()
            car_ads_df = pd.concat([car_ads_df, cargurus_df], ignore_index=True)

        logger.info(f"Found {len(car_ads_df)} car ads.")

        self.df = car_ads_df

    def preprocess_ads(self, top_n_options: int = 50):
        # determine age of ad at posting
        start_time = pd.Timestamp.now()
        self.df["age_at_posting"] = self.df.listed_date.dt.year - self.df.year

        # calculate mileage per year
        self.df["mileage_per_year"] = self.df.mileage / self.df.age_at_posting

        # set new vechicle mileage_per_year to mileage
        self.df.loc[self.df.age_at_posting <= 0, "mileage_per_year"] = self.df.loc[
            self.df.age_at_posting <= 0, "mileage"
        ]

        # for any columns of dtype categorical cast to type strnig for modifications
        for col in self.df.select_dtypes(include="category").columns:
            self.df[col] = self.df[col].astype(str)

        model_correction_map = {
            "F 150": "F-150",
            "RAV 4": "RAV4",
            "F 150 Raptor": "F-150 Raptor",
        }

        self.df.model = self.df.model.replace(model_correction_map)

        make_correction_map = {
            "INFINITI": "Infiniti",
            "FIAT": "Fiat",
        }

        self.df.make = self.df.make.replace(make_correction_map)

        # normalize Dodge RAM to RAM as per 2009 make name change from all dodge trucks to RAM
        self.df.loc[
            (self.df.make == "Dodge") & (self.df.model.str.lower().str.contains("ram")),
            "make",
        ] = "RAM"

        # fix RAM model names where RAM is still in the model name, remove it and extra whitespace
        self.df.loc[
            (self.df.make == "RAM") & (self.df.model.str.lower().str.contains("ram")),
            "model",
        ] = (
            self.df.loc[
                (self.df.make == "RAM")
                & (self.df.model.str.lower().str.contains("ram")),
                "model",
            ]
            .str.lower()
            .str.replace("ram", "")
            .str.strip()
            .str.title()
        )

        # pre process options list for one hot encoding using multilabel binarizer
        self.get_car_options(top_n_options=top_n_options)

        logger.info(
            f"Done preprocessing car ads, took {(pd.Timestamp.now() - start_time).seconds}s."
        )

    def _get_cargurus_ads(self) -> pd.DataFrame:
        """Gets all cargurus car ads.

        Returns
        -------
        pd.DataFrame
            A dataframe containing all cargurus car ads.
        """

        columns_to_load = [
            "make",
            "model",
            "year",
            "listed_date",
            "price",
            "mileage",
            "major_options",
            "seller_rating",
            "horsepower",
            "fuel_type",
            "wheel_system",
            "currency",
            "exchange_rate_usd_to_cad",
            "is_new",
        ]

        parquet_filter = [
            ("year", ">=", self.year_range[0]),
            ("year", "<=", self.year_range[1]),
        ]

        if self.make:
            parquet_filter.append(("make", "==", self.make))

        if self.model:
            parquet_filter.append(("model", "==", self.model))

        logger.info("Getting all cargurus car ads...")
        cargurus_df = pd.read_parquet(
            os.path.join(
                SRC_PATH, "data", "processed", "processed-cargurus-ads.parquet"
            ),
            # filter by year_range and make and model
            filters=parquet_filter,
            columns=columns_to_load,
        )

        cargurus_df["source"] = "cargurus"

        cargurus_df["condition"] = "Used"
        # for values where is_new column is true, set condition to new
        cargurus_df.loc[cargurus_df.is_new, "condition"] = "New"
        # drop is_new column
        cargurus_df.drop(columns=["is_new"], inplace=True)

        logger.info(f"Found {len(cargurus_df)} cargurus car ads.")

        return cargurus_df

    def _get_kijiji_ads(self, limit_ads: int = None) -> pd.DataFrame:
        """Gets all kijiji car ads.

        Returns
        -------
        pd.DataFrame
            A dataframe containing all kijiji car ads.
        """

        client, db, collection = connect_to_database()

        logger.info("Getting all kijiji car ads...")
        # query collection with year_range and make and model if they exist
        query = {
            "year": {"$gte": self.year_range[0], "$lte": self.year_range[1]},
        }

        # define projection to only return the columns we need
        projection = {
            "year": 1,
            "make": 1,
            "model": 1,
            "features": 1,
            "created": 1,
            "modified": 1,
            "price": 1,
            "driveTrain": 1,
            "condition": 1,
            "mileage": 1,
            "url": 1,
            "transmission": 1,
            "location": 1,
            "_id": 0,
        }

        # add filter for make if it exists
        if self.make:
            query["make"] = self.make

        # add filter for model if it exists
        if self.model:
            query["model"] = self.model

        if limit_ads is not None:
            kijiji_df = pd.DataFrame(
                list(collection.find(query, projection).limit(limit_ads))
            )
        else:
            kijiji_df = pd.DataFrame(list(collection.find(query, projection)))

        kijiji_df["source"] = "kijiji"

        # convert created column from unix seconds since epoch to datetime
        kijiji_df["listed_date"] = pd.to_datetime(kijiji_df.created, unit="s")
        drive_train_map = defaultdict(lambda: "Unknown")
        drive_train_map.update(
            {
                "Front-wheel drive (FWD)": "FWD",
                "Rear-wheel drive (RWD)": "RWD",
                "Four-wheel drive": "4WD",
                "All-wheel drive (AWD)": "AWD",
            }
        )

        # mpa and replace driveTrain with wheel_system values
        kijiji_df["wheel_system"] = kijiji_df.driveTrain.map(
            drive_train_map,
            na_action="ignore",
        )

        # where condition value is a dict extract the condition value
        kijiji_df["condition"] = kijiji_df.condition.apply(
            lambda x: x["condition"] if isinstance(x, dict) else x
        )

        # some locations ore lists of dicts for location, extract the first location from stateProvince field
        kijiji_df["province"] = kijiji_df.location.apply(
            lambda x: (
                x["stateProvince"]
                if isinstance(x, dict)
                else x[0]["stateProvince"] if isinstance(x, list) else x
            )
        )

        logger.info(f"Found {len(kijiji_df)} kijiji car ads.")

        return kijiji_df

    def find_make_model_names(self, make=None) -> list:
        """Finds all models for a given make.

        Parameters
        ----------
        make : str
            The make to find models for.

        Returns
        -------
        list
            A list of models for the given make.
        """

        # get all makes and models
        all_makes_models_df = pd.read_json(
            os.path.join(SRC_PATH, "data", "processed", "all-makes-models.json"),
            orient="index",
        )

        # if make is None, return all makes
        if make is None:
            all_makes = pd.DataFrame(
                {"Makes": [", ".join(all_makes_models_df.columns.to_list())]}
            )

            all_makes_styled = (
                all_makes[["Makes"]]
                .style.hide(axis="index")
                .set_properties(
                    **{
                        "inline-size": "700px",
                        "overflow-wrap": "break-word",
                        "text-align": "center",
                        "font-weight": "bold",
                        "font-size": "15px",
                    },
                )
                .set_table_styles(
                    [dict(selector="th", props=[("text-align", "center")])]
                )
            )

            return all_makes_styled

        if make not in all_makes_models_df.columns:
            raise ValueError(f"No make named '{make}' found.")

        # get all models and style output to make it readable
        models_styled = (
            all_makes_models_df[[make]]
            .style.set_properties(
                **{
                    "inline-size": "500px",
                    "overflow-wrap": "break-word",
                    "text-align": "center",
                    "font-weight": "bold",
                    "font-size": "15px",
                },
            )
            .set_table_styles(  # centers the table columns
                [dict(selector="th", props=[("text-align", "center")])]
            )
        )

        return models_styled

    def export_makes_model_names(self, output_path: str = None) -> None:
        """Exports all makes and models to a json file.

        Raises
        ------
        ValueError
            If no car ads have been loaded.
        """

        if output_path is None:
            output_path = os.path.join(
                SRC_PATH, "data", "processed", "all-makes-models.json"
            )

        if self.df is None:
            raise ValueError("No car ads have been loaded.")

        # get all makes and models
        make_model_dict = dict()
        for s in self.df.source.unique():
            make_model_dict[s] = dict()
            source_df = self.df.query(f"source == '{s}'")
            for make in source_df.make.unique():
                make_model_dict[s][make] = (
                    source_df.query(f"make == '{make}'").model.unique().tolist()
                )

        # export the dictionary to a json file
        with open(
            output_path,
            "w",
        ) as f:
            json.dump(make_model_dict, f)

    def export_to_parquet(self, path: str) -> None:
        """Exports the dataframe to a parquet file.

        Parameters
        ----------
        path : str
            The path to the parquet file.
        """

        if self.df is None:
            raise ValueError("No car ads have been loaded to export.")
        if len(self.df) == 0:
            raise ValueError("The dataframe contains no rows to export.")

        logger.info("Exporting dataframe to parquet file...")
        self.df.to_parquet(path, index=False, engine="pyarrow")

    def export_to_csv(self, path: str) -> None:
        """Exports the dataframe to a csv file.

        Parameters
        ----------
        path : str
            The path to the csv file.
        """
        if self.df is None:
            raise ValueError("No car ads have been loaded to export.")
        if len(self.df) == 0:
            raise ValueError("The dataframe contains no rows to export.")

        logger.info("Exporting dataframe to csv file...")
        self.df.to_csv(path)

    def get_car_options(self, top_n_options: int = 50):
        """
        Get the car options from the major_options column in the ads dataframe
        and return a dataframe with the options one hot encoded
        """

        if self.df is None:
            raise ValueError("No car ads have been loaded.")

        self.df["options_list"] = None

        if "cargurus" in self.sources:
            # parse cargurus strings of options into list of options
            self.df.loc[self.df.source == "cargurus", "options_list"] = (
                self.df.loc[self.df.source == "cargurus", "major_options"]
                .str.strip("['']")
                .str.replace("'", "")
                .str.replace(", ", ",")
                .str.replace(" ", "-")
                .str.replace("/", "-")
                .str.lower()
                .str.split(",")
            )

        if "kijiji" in self.sources:
            # if the options_list for kijiji ads are lists, replace ' with "", replace / with -,
            # make it lowercase, replace spaces with -
            if (
                self.df.loc[self.df.source == "kijiji", "features"]
                .apply(lambda x: isinstance(x, list))
                .any()
            ):
                # for each entry in the list apply the formatting
                self.df.loc[self.df.source == "kijiji", "options_list"] = self.df.loc[
                    self.df.source == "kijiji", "features"
                ].apply(
                    lambda x: (
                        [
                            option.replace("'", "")
                            .replace("/", "-")
                            .lower()
                            .replace(" ", "-")
                            for option in x
                            if isinstance(x, list)
                        ]
                        if isinstance(x, list)
                        else x
                    )  # If x is not a list, keep the original value
                )
            # otherwise assumed major_options is a string
            elif (
                self.df.loc[self.df.source == "kijiji", "features"]
                .apply(lambda x: isinstance(x, str))
                .any()
            ):
                # need to add option for lists when reading from cosmos database
                self.df.loc[self.df.source == "kijiji", "options_list"] = (
                    self.df.loc[self.df.source == "kijiji", "features"]
                    .str.strip("['']")
                    .str.replace("'", "")
                    .str.replace(", ", ",")
                    .str.replace(" ", "-")
                    .str.replace("/", "-")
                    .str.lower()
                    .str.split(",")
                )
            else:
                logger.error(
                    f"kijiji ads features column not type str or list, no parsing done."
                )

        # reset the index to use as uniqwue id's for each ad as cargurus doesn't have unique id's
        if "unique_id" not in self.df.columns:
            self.df = self.df.reset_index().rename(columns={"index": "unique_id"})

        self.df = self.df.explode("options_list")

        # rename identical options
        options_combo_map = {
            "apple-carplay": "apple-carplay-android-auto",
            "carplay": "apple-carplay-android-auto",
            "android-auto": "apple-carplay-android-auto",
            "a-c-(automatic)": "air-conditioning",
            "a-c-(2-zones)": "air-conditioning",
            "electric-heated-seats": "heated-seats",
            "blind-spot-assist": "blind-spot-monitoring",
            "sunroof": "sunroof-moonroof",
            "adaptive-cruise-control": "cruise-control",
        }

        self.df.options_list = self.df.options_list.replace(options_combo_map)

        # only keep the top n options by count
        most_common_options = self.df.options_list.value_counts()[
            :top_n_options
        ].index.to_list()

        logger.debug(
            f"Keeping the top {top_n_options} options by count: {most_common_options}"
        )

        # drop all rows where options_list is not in most_common_options but keep empty rows
        self.df = self.df[
            (self.df.options_list.isin(most_common_options))
            | (self.df.options_list.isna())
        ]

        # replace na values with 'none-listed'
        self.df.options_list = self.df.options_list.fillna("none-listed")

        # get the one hot encoded options for each ad by grouping opttions_list column
        self.df = (
            self.df.groupby(by="unique_id", as_index=False)
            .agg(
                {
                    "options_list": list,
                    **{
                        col: "first" for col in self.df.columns if col != "options_list"
                    },
                }
            )
            .reset_index(drop=True)
        )

        logger.info(
            f"Vehicle option preprocessing complete, kept top {top_n_options} options by count."
        )
