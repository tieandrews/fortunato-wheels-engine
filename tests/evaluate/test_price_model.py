# Author: Ty Andrews
# Date: 2023-08-08

import os, sys

import pytest
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

SRC_PATH = os.path.join(os.path.dirname(__file__), "..", "..")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from src.evaluate.price_model import (
    calculate_evaluation_metrics, 
    calculate_evaluation_metrics_by_model,
    calculate_evaluation_metrics_by_make
)

# Sample fixture for generating test data
@pytest.fixture
def sample_data():
    np.random.seed(42)
    n_samples = 100
    y_true = np.random.rand(n_samples)
    y_pred = y_true + np.random.normal(0, 0.1, n_samples)
    return y_true, y_pred

# Test cases
def test_rmse(sample_data):
    y_true, y_pred = sample_data
    metrics_df = calculate_evaluation_metrics(y_true, y_pred, metrics=["rmse"])
    assert metrics_df.shape == (1, 2)
    assert metrics_df.loc[0, "metric"] == "rmse"
    assert metrics_df.loc[0, "value"] == round(np.sqrt(mean_squared_error(y_true, y_pred)), 1)

def test_mape(sample_data):
    y_true, y_pred = sample_data
    metrics_df = calculate_evaluation_metrics(y_true, y_pred, metrics=["mape"])
    assert metrics_df.shape == (1, 2)
    assert metrics_df.loc[0, "metric"] == "mape"
    assert metrics_df.loc[0, "value"] == round((np.abs(y_true - y_pred) / y_true).mean(), 4)

def test_r2(sample_data):
    y_true, y_pred = sample_data
    metrics_df = calculate_evaluation_metrics(y_true, y_pred, metrics=["r2"])
    assert metrics_df.shape == (1, 2)
    assert metrics_df.loc[0, "metric"] == "r2"
    assert metrics_df.loc[0, "value"] == round(r2_score(y_true, y_pred), 4)

def test_multiple_metrics(sample_data):
    y_true, y_pred = sample_data
    metrics_df = calculate_evaluation_metrics(y_true, y_pred, metrics=["rmse", "mape", "r2"])
    assert metrics_df.shape == (3, 2)
    assert set(metrics_df["metric"]) == {"rmse", "mape", "r2"}

def test_invalid_metric(sample_data):
    y_true, y_pred = sample_data
    with pytest.raises(ValueError):
        calculate_evaluation_metrics(y_true, y_pred, metrics=["invalid_metric"])

def test_empty_data():
    with pytest.raises(ValueError):
        calculate_evaluation_metrics([], [], metrics=["rmse"])

def test_empty_metrics(sample_data):
    y_true, y_pred = sample_data
    with pytest.raises(ValueError):
        metrics_df = calculate_evaluation_metrics(y_true, y_pred, metrics=[])
    

# Sample fixture for generating test data
@pytest.fixture
def sample_data_by_model():
    data = {
        'make': ['Toyota', 'Toyota', 'Ford', 'Ford', 'Honda', 'Honda'],
        'model': ['Camry', 'Corolla', 'Mustang', 'Fiesta', 'Civic', 'Civic'],
        'price': [20000, 18000, 25000, 15000, 22000, 21000],
        'predicted_price': [19800, 18090, 25500, 14500, 22500, 20800]
    }
    return pd.DataFrame(data)

# TESTING METRICS BY MODEL
def test_rmse_by_model(sample_data_by_model):
    df = sample_data_by_model
    metrics_df = calculate_evaluation_metrics_by_model(df, metrics=["rmse"])
    
    assert metrics_df.shape == (5, 4)
    assert metrics_df.loc[0, "make"] == "Toyota"
    assert metrics_df.loc[0, "model"] == "Camry"
    assert metrics_df.loc[0, "RMSE"] == 200.0

def test_mape_by_model(sample_data_by_model):
    df = sample_data_by_model
    metrics_df = calculate_evaluation_metrics_by_model(df, metrics=["mape"])
    
    assert metrics_df.shape == (5, 4)
    assert metrics_df.loc[1, "make"] == "Toyota"
    assert metrics_df.loc[1, "model"] == "Corolla"
    assert metrics_df.loc[1, "MAPE"] == 0.005

def test_r2_by_model(sample_data_by_model):
    df = sample_data_by_model
    metrics_df = calculate_evaluation_metrics_by_model(df, metrics=["r2"])
    
    assert metrics_df.shape == (5, 4)
    assert metrics_df.loc[4, "make"] == "Honda"
    assert metrics_df.loc[4, "model"] == "Civic"
    assert metrics_df.loc[4, "R2"] == round(r2_score([22000, 21000], [22500, 20800]), 4)

def test_multiple_metrics_by_model(sample_data_by_model):
    df = sample_data_by_model
    metrics_df = calculate_evaluation_metrics_by_model(df, metrics=["rmse", "mape", "r2"])
    
    assert metrics_df.shape == (5, 6)
    assert set(metrics_df.columns) == {"make", "model", "count", "RMSE", "MAPE", "R2"}

def test_missing_columns():
    with pytest.raises(ValueError):
        calculate_evaluation_metrics_by_model(pd.DataFrame({'make': [], 'model': [], 'price': [], 'another_column': []}))

# TESTING METRICS BY MAKE

def test_rmse_by_make(sample_data_by_model):
    df = sample_data_by_model
    metrics_df = calculate_evaluation_metrics_by_make(df, metrics=["rmse"])
    
    assert metrics_df.shape == (3, 3)
    assert metrics_df.loc[0, "make"] == "Toyota"
    assert metrics_df.loc[0, "RMSE"] == round(np.sqrt(mean_squared_error([20000, 18000], [19800, 18090])), 1)

def test_mape_by_make(sample_data_by_model):
    df = sample_data_by_model
    metrics_df = calculate_evaluation_metrics_by_make(df, metrics=["mape"])
    
    assert metrics_df.shape == (3, 3)
    assert metrics_df.loc[1, "make"] == "Ford"
    
    actual_prices = np.array([25000, 15000])
    predicted_prices = np.array([25500, 14500])
    mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices))
    
    assert metrics_df.loc[1, "MAPE"] == round(mape, 4)

def test_r2_by_make(sample_data_by_model):
    df = sample_data_by_model
    metrics_df = calculate_evaluation_metrics_by_make(df, metrics=["r2"])
    
    assert metrics_df.shape == (3, 3)
    assert metrics_df.loc[2, "make"] == "Honda"
    assert metrics_df.loc[2, "R2"] == round(r2_score([22000, 21000], [22500, 20800]), 4)

def test_multiple_metrics_by_make(sample_data_by_model):
    df = sample_data_by_model
    metrics_df = calculate_evaluation_metrics_by_make(df, metrics=["rmse", "mape", "r2"])
    
    assert metrics_df.shape == (3, 5)
    assert set(metrics_df.columns) == {"make", "count", "RMSE", "MAPE", "R2"}

def test_missing_columns():
    with pytest.raises(ValueError):
        calculate_evaluation_metrics_by_make(pd.DataFrame({'make': [], 'price': [], 'another_column': []}))
