# Author: Ty Andrews
# Date: 2023-07-21

import os
import sys

import pytest
import pandas as pd
import numpy as np
import datetime as dt
import json

SRC_PATH = os.path.join(os.path.dirname(__file__), "..", "..")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from src.data.car_ads import CarAds


@pytest.fixture
def empty_car_ads():
    return CarAds()


@pytest.fixture
def raw_kj_car_ad():

    raw_kj_car_ad = CarAds()
    raw_kj_car_ad.sources = ["kijiji"]
    raw_kj_car_ad.df = pd.DataFrame(
        {
            "source": ["kijiji"],
            "id": [1],
            "price": [10_000],
            "year": [2020],
            "mileage": [10_000],
            "make": ["Honda"],
            "model": ["CR-V"],
            "features": [["Air Conditioning", "Backup Camera"]],
            "major_options": [""],
            "listed_date": [pd.to_datetime("2022-01-01")],
        }
    )

    return raw_kj_car_ad


@pytest.fixture
def raw_cargurus_car_ad():

    raw_cargurus_car_ad = CarAds()
    raw_cargurus_car_ad.sources = ["cargurus"]
    raw_cargurus_car_ad.df = pd.DataFrame(
        {
            "source": ["cargurus"],
            "id": [1],
            "price": [5_000],
            "year": [2017],
            "mileage": [12_000],
            "make": ["Toyota"],
            "model": ["RAV4"],
            "features": [[]],
            "major_options": ["Heated Seats"],
            "listed_date": [pd.to_datetime("2020-01-01")],
        }
    )

    return raw_cargurus_car_ad


@pytest.fixture
def empty_car_ads():
    return CarAds()


# test that a data dump not in parquet or CSV format raises an error
def test_load_data_dump(empty_car_ads):

    with pytest.raises(ValueError):
        empty_car_ads.get_car_ads(data_dump="invalid_file_type.json")


# test that the parsing of car options works as expected
def test_parse_car_options(raw_kj_car_ad, raw_cargurus_car_ad):

    raw_kj_car_ad.preprocess_ads()
    raw_cargurus_car_ad.preprocess_ads()

    raw_kj_car_ad.get_car_options()
    raw_cargurus_car_ad.get_car_options()

    assert raw_kj_car_ad.df.options_list.iloc[0] == [
        "air-conditioning",
        "backup-camera",
    ]
    assert raw_cargurus_car_ad.df.options_list.iloc[0] == ["heated-seats"]


# test that preprocessing a raw kijiji ad returns the expected processed ad
def test_preprocess_raw_kijiji_ad(raw_kj_car_ad, raw_cargurus_car_ad):

    raw_kj_car_ad.preprocess_ads()
    raw_cargurus_car_ad.preprocess_ads()

    assert raw_kj_car_ad.df.age_at_posting.iloc[0] == 2
    assert raw_kj_car_ad.df.mileage_per_year.iloc[0] == 5_000
    assert raw_kj_car_ad.df.options_list.iloc[0] == [
        "air-conditioning",
        "backup-camera",
    ]

    assert raw_cargurus_car_ad.df.age_at_posting.iloc[0] == 3
    assert raw_cargurus_car_ad.df.mileage_per_year.iloc[0] == 4_000
    assert raw_cargurus_car_ad.df.options_list.iloc[0] == ["heated-seats"]


def test_preprocess_ads_age_zero(empty_car_ads):

    empty_car_ads.df = pd.DataFrame(
        {
            "source": ["kijiji"],
            "year": [2021],
            "make": ["Ford"],
            "model": ["F-150"],
            "mileage": [100000],
            "listed_date": [dt.datetime(2021, 1, 1, 0, 0, 0)],
            "major_options": [""],
            "features": [[]],
        }
    )
    empty_car_ads.preprocess_ads()
    assert empty_car_ads.df.loc[0, "age_at_posting"] == 0
    assert empty_car_ads.df.loc[0, "mileage_per_year"] == 100000


def test_find_make_names(empty_car_ads):

    make_options = empty_car_ads.find_make_model_names()

    # check that the make options are a styled object
    assert isinstance(make_options, pd.io.formats.style.Styler)
    # check that the most common vehicle makes are present
    assert "Honda" in make_options.data.Makes.iloc[0]
    assert "Ford" in make_options.data.Makes.iloc[0]
    assert "Jeep" in make_options.data.Makes.iloc[0]


def test_select_single_make(empty_car_ads):

    model_options = empty_car_ads.find_make_model_names(make="Honda")

    # check the model options are a styled object
    assert isinstance(model_options, pd.io.formats.style.Styler)
    # check that the most common Honda models are present
    assert "Civic" in model_options.data.Honda.iloc[0]
    assert "CR-V" in model_options.data.Honda.iloc[0]
    assert "Accord" in model_options.data.Honda.iloc[0]


def test_non_existant_make(empty_car_ads):

    with pytest.raises(ValueError):
        empty_car_ads.find_make_model_names(make="invalid_make")


# Test the export_to_parquet function
def test_export_to_parquet(raw_kj_car_ad, tmp_path):
    output_path = os.path.join(tmp_path, "test_export.parquet")
    raw_kj_car_ad.export_to_parquet(output_path)
    assert os.path.exists(output_path)


# Test the export_to_csv function
def test_export_to_csv(raw_kj_car_ad, tmp_path):
    output_path = os.path.join(tmp_path, "test_export.csv")
    raw_kj_car_ad.export_to_csv(output_path)
    assert os.path.exists(output_path)


# Test edge case when no data is provided for export_to_parquet
def test_export_to_parquet_empty_data(tmp_path):
    empty_car_class = CarAds()
    output_path = os.path.join(tmp_path, "empty_test_export.parquet")
    with pytest.raises(ValueError, match="No car ads have been loaded."):
        empty_car_class.export_to_parquet(output_path)


# Test edge case when no data is provided for export_to_csv
def test_export_to_csv_empty_data(tmp_path):
    empty_car_class = CarAds()
    output_path = os.path.join(tmp_path, "empty_test_export.csv")
    with pytest.raises(ValueError, match="No car ads have been loaded."):
        empty_car_class.export_to_csv(output_path)


# test that export make_makes_model_names works
def test_export_make_model_names(raw_kj_car_ad, tmp_path):
    output_path = os.path.join(tmp_path, "test_export_make_model_names.json")
    raw_kj_car_ad.export_makes_model_names(output_path)
    assert os.path.exists(output_path)

    # check that loading the json file works
    with open(output_path, "r") as f:
        make_model_names = json.load(f)

    assert "Honda" in make_model_names["kijiji"].keys()
    assert "CR-V" in make_model_names["kijiji"]["Honda"]
