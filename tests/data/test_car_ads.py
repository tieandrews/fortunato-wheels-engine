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
def simple_data():
    data = {'year': [2010, 2011, 2012, 2013, 2014],
            'make': ['Ford', 'Toyota', 'Honda', 'Ford', 'Toyota'],
            'model': ['F-150', 'RAV4', 'Civic', 'Mustang', 'Camry'],
            'mileage': [100000, 80000, 60000, 40000, 20000],
            'listed_date' : [dt.datetime(2021, 1, 1, 0, 0, 0), dt.datetime(2021, 1, 2, 0, 0, 0), dt.datetime(2021, 1, 3, 0, 0, 0), dt.datetime(2021, 1, 4, 0, 0, 0), dt.datetime(2021, 1, 5, 0, 0, 0)]
        }
    return pd.DataFrame(data)
    
@pytest.fixture
def car_ads():
    return CarAds()

# check the age_at_posting and mileage_per_year columns are created correctly for basic case   
def test_preprocess_ads(simple_data, car_ads):
    car_ads.df = simple_data
    car_ads.preprocess_ads()
    for _, row in simple_data.iterrows():
        age_at_posting = row.listed_date.year - row.year
        mileage_per_year = row.mileage / age_at_posting
        assert car_ads.df.loc[row.name, 'age_at_posting'] == age_at_posting
        assert car_ads.df.loc[row.name, 'mileage_per_year'] == mileage_per_year

# test that a car with an age at zero has mileage per year set to it's current mileage
def test_preprocess_ads_age_zero(car_ads):

    car_ads.df = pd.DataFrame({'year': [2021],
                               'make': ['Ford'],
                               'model': ['F-150'],
                               'mileage': [100000],
                               'listed_date' : [dt.datetime(2021, 1, 1, 0, 0, 0)]
                            })
    car_ads.preprocess_ads()
    assert car_ads.df.loc[0, 'age_at_posting'] == 0
    assert car_ads.df.loc[0, 'mileage_per_year'] == 100000

def test_find_make_model_names(car_ads):
    # convert to string to check content instead of Pandas styler object
    car_makes_result_string = car_ads.find_make_model_names().to_string()

    # check that a handful of common makes are in the results
    assert 'Ford' in car_makes_result_string
    assert 'Toyota' in car_makes_result_string
    assert 'Honda' in car_makes_result_string
    assert 'Chevrolet' in car_makes_result_string
    assert 'Nissan' in car_makes_result_string
    assert 'Hyundai' in car_makes_result_string

def test_find_make_model_names_nonexistent_make():

    pytest.raises(ValueError, CarAds().find_make_model_names, 'NotAMake')


# Sample car ad data for testing
SAMPLE_CAR_ADS = {
    "source": ["source1", "source1", "source2"],
    "make": ["Toyota", "Toyota", "Honda"],
    "model": ["Camry", "Corolla", "Civic"],
}

# Fixture to create a sample CarClass instance with test data
@pytest.fixture
def car_class_instance(car_ads, tmp_path):
    # Create a DataFrame with sample data
    df = pd.DataFrame(SAMPLE_CAR_ADS)
    car_class = car_ads
    car_class.df = df
    car_class.export_to_parquet(os.path.join(tmp_path, "test_data.parquet"))
    return car_class


# Test the export_makes_model_names function
def test_export_makes_model_names(car_class_instance, tmpdir):
    # Test exporting when car ads are loaded
    output_path = os.path.join(tmpdir, "all-makes-models.json")
    car_class_instance.export_makes_model_names(output_path=output_path)
    
    # assert len(tmpdir.listdir()) == 1
    assert os.path.exists(output_path)

    # Verify the content of the exported json file
    with open(output_path, "r") as f:
        make_model_dict = json.load(f)

    assert make_model_dict == {
        "source1": {"Toyota": ["Camry", "Corolla"]},
        "source2": {"Honda": ["Civic"]},
    }

    # Test exporting when no car ads are loaded
    empty_car_class = CarAds()
    with pytest.raises(ValueError, match="No car ads have been loaded."):
        empty_car_class.export_makes_model_names()


# Test the export_to_parquet function
def test_export_to_parquet(car_class_instance, tmp_path):
    output_path = os.path.join(tmp_path, "test_export.parquet")
    car_class_instance.export_to_parquet(output_path)
    assert os.path.exists(output_path)

    # Verify the content of the exported parquet file (optional)
    df = pd.read_parquet(output_path)
    assert df.equals(car_class_instance.df)


# Test the export_to_csv function
def test_export_to_csv(car_class_instance, tmp_path):
    output_path = os.path.join(tmp_path, "test_export.csv")
    car_class_instance.export_to_csv(output_path)
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