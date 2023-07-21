# Author: Ty Andrews
# Date: 2023-04-05
import os
import sys

import logging
import pandas as pd
import numpy as np
import json
from collections import defaultdict

cur_dir = os.getcwd()
SRC_PATH = cur_dir[
    : cur_dir.index("fortunato-wheels-engine") + len("fortunato-wheels-engine")
]
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from src.data.upload_to_db import connect_to_database
from src.logs import get_logger

# Create a custom logger
logger = get_logger(__name__)


class CarAds:
    def __init__(self):
        self.year_range = None
        self.make = None
        self.model = None
        self.df = None

    def get_car_ads(
        self,
        year_range: tuple = None,
        make: str = None,
        model: str = None,
        sources: list = ["cargurus", "kijiji"],
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

        if data_dump is not None:
            logger.info(f"Loading car ads from {data_dump}...")

            if data_dump.endswith(".csv"):
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
        if "cargurus" in sources:
            # get cargurus ads
            cargurus_df = self._get_cargurus_ads()
            car_ads_df = pd.concat([car_ads_df, cargurus_df], ignore_index=True)

        if "kijiji" in sources:
            # get kijiji ads
            kijiji_df = self._get_kijiji_ads()
            car_ads_df = pd.concat([car_ads_df, kijiji_df], ignore_index=True)

        logger.info(f"Found {len(car_ads_df)} car ads.")

        self.df = car_ads_df

    def preprocess_ads(self):
        # determine age of ad at posting
        self.df["age_at_posting"] = self.df.listed_date.dt.year - self.df.year

        # calculate mileage per year
        self.df["mileage_per_year"] = self.df.mileage / self.df.age_at_posting

        self.df.mileage_per_year = self.df.mileage_per_year.replace(
            [np.inf, -np.inf], np.nan
        )

        model_correction_map = {
            "F 150": "F-150",
            "RAV 4": "RAV4",
            "F 150 Raptor": "F-150 Raptor",
        }

        self.df.model = self.df.model.replace(model_correction_map)


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

        logger.info(f"Found {len(cargurus_df)} cargurus car ads.")

        return cargurus_df

    def _get_kijiji_ads(self) -> pd.DataFrame:
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

        # add filter for make if it exists
        if self.make:
            query["make"] = self.make

        # add filter for model if it exists
        if self.model:
            query["model"] = self.model

        kijiji_df = pd.DataFrame(list(collection.find(query)))

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
            os.path.join(
                SRC_PATH, "data", "scraping-tracking", "all-makes-models.json"
            ),
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

    def export_makes_model_names(self):
        """Exports all makes and models to a json file.

        Raises
        ------
        ValueError
            If no car ads have been loaded.
        """

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
            os.path.join(
                SRC_PATH, "data", "scraping-tracking", "all-makes-models.json"
            ),
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
        logger.info("Exporting dataframe to parquet file...")
        self.df.to_parquet(path, index=False, engine="pyarrow")

    def export_to_csv(self, path: str) -> None:
        """Exports the dataframe to a csv file.

        Parameters
        ----------
        path : str
            The path to the csv file.
        """
        logger.info("Exporting dataframe to csv file...")
        self.df.to_csv(path)
