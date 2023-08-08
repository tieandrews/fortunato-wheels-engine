# Author: Ty Andrews
# date: 2023-08-04

import os, sys

import pytest
import pandas as pd

cur_dir = os.getcwd()
SRC_PATH = cur_dir[
    : cur_dir.index("fortunato-wheels-engine") + len("fortunato-wheels-engine")
]
print(f"SRC_PATH: {SRC_PATH}")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from src.data.car_ads import CarAds

@pytest.fixture
def empty_car_ads():
    return CarAds()

@pytest.fixture
def raw_kj_car_ad():

    raw_kj_car_ad = CarAds()
    raw_kj_car_ad.df = pd.DataFrame({
        'source': ['kijiji'],
        'id': [1],
        'price': [10_000],
        'year': [2020],
        'mileage': [10_000],
        'make': ['Honda'],
        'model': ['CR-V'],
        'features': [["Air Conditioning", "Backup Camera"]],
        "major_options": [""],
        'listed_date': [pd.to_datetime('2022-01-01')],
    })

    return raw_kj_car_ad

@pytest.fixture
def raw_cargurus_car_ad():

    raw_cargurus_car_ad = CarAds()
    raw_cargurus_car_ad.df = pd.DataFrame({
        'source': ['cargurus'],
        'id': [1],
        'price': [12_000],
        'year': [2017],
        'mileage': [10_000],
        'make': ['Toyota'],
        'model': ['RAV4'],
        'features': [["Heated Seats"]],
        "major_options": [""],
        'listed_date': [pd.to_datetime('2020-01-01')],
    })

    return raw_cargurus_car_ad

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

    assert raw_kj_car_ad.df.options_list.iloc[0] == ["air-conditioning", "backup-camera"]
    assert raw_cargurus_car_ad.df.options_list.iloc[0] == ["heated-seats"]

# test that preprocessing a raw kijiji ad returns the expected processed ad
def test_preprocess_raw_kijiji_ad(raw_kj_car_ad, raw_cargurus_car_ad):

    raw_kj_car_ad.preprocess_ads()
    raw_cargurus_car_ad.preprocess_ads()

    assert raw_kj_car_ad.df.age_at_posting.iloc[0] == 2
    assert raw_kj_car_ad.df.mileage_per_year.iloc[0] == 5_000
    assert raw_kj_car_ad.df.options_list.iloc[0] == ["air-conditioning", "backup-camera"]

    assert raw_cargurus_car_ad.df.age_at_posting.iloc[0] == 3
    assert raw_cargurus_car_ad.df.mileage_per_year.iloc[0] == 4_000
    assert raw_cargurus_car_ad.df.options_list.iloc[0] == ["heated-seats"]


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

# test that exporting the car_ads.df to csv works using a temporary file
def test_export_to_csv(raw_kj_car_ad):
    
        raw_kj_car_ad.preprocess_ads()
    
        raw_kj_car_ad.export_to_csv("test_export.csv")
    
        assert os.path.exists("test_export.csv")
    
        os.remove("test_export.csv")

# test that exporting the car_ads.df to parquet works using a temporary file
def test_export_to_parquet(raw_kj_car_ad):
        
    raw_kj_car_ad.preprocess_ads()

    raw_kj_car_ad.export_to_parquet("test_export.parquet")

    assert os.path.exists("test_export.parquet")

    os.remove("test_export.parquet")

