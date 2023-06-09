# Author: Ty Andrews
# Date: 2023-06-04
import os, sys

import pandas as pd
from azureml.core.model import Model
import joblib
import json
import numpy as np
from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.standard_py_parameter_type import (
    StandardPythonParameterType,
)
from azureml.core import Workspace
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def init():
    global price_model, price_quant5_model, price_quant95_model

    price_model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"),
        "all-vehicles-price-prediction",
        "3",
        "model",
        "model.pkl",
    )

    price_quant5_model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"),
        "all-vehicles-price-prediction-quant5",
        "1",
        "model_q5",
        "model.pkl",
    )

    price_quant95_model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"),
        "all-vehicles-price-prediction-quant95",
        "1",
        "model_q95",
        "model.pkl",
    )

    logger.info("Loaded model from " + price_model_path)

    # Deserialize the model files back into scikit-learn models.
    price_model = joblib.load(price_model_path)
    price_quant5_model = joblib.load(price_quant5_model_path)
    price_quant95_model = joblib.load(price_quant95_model_path)


sample_input = StandardPythonParameterType(
    {
        "data": {
            "vehicle_data": {
                "model": StandardPythonParameterType("rav4"),
                "age_at_posting": StandardPythonParameterType(2018),
                "mileage_per_year": StandardPythonParameterType(10000),
                "make": StandardPythonParameterType("toyota"),
                "wheel_system": StandardPythonParameterType("4wd"),
            }
        }
    }
)
sample_output = StandardPythonParameterType(
    {
        "predicted_price": StandardPythonParameterType(23_000),
        "upper_ci": StandardPythonParameterType(25_000),
        "lower_ci": StandardPythonParameterType(21_000),
    }
)


@input_schema("request_data", sample_input)
@output_schema(sample_output)
def run(request_data):
    try:
        data_dict = request_data["data"]["vehicle_data"]

        vehicle_model = data_dict["model"]
        age_at_posting = data_dict["age_at_posting"]
        mileage_per_year = data_dict["mileage_per_year"]
        make = data_dict["make"]
        if "wheel_system" in data_dict.keys():
            wheel_system = data_dict["wheel_system"]
        else:
            wheel_system = "fwd"

        # assemble the features into a dataframe
        vehicle_features = pd.DataFrame(
            {
                "model": [vehicle_model],
                "age_at_posting": [age_at_posting],
                "mileage_per_year": [mileage_per_year],
                "make": [make],
                "wheel_system": [wheel_system],
            }
        )

        # Call predict() on each model
        price_pred = price_model.predict(vehicle_features)
        upper_ci = price_quant95_model.predict(vehicle_features)
        lower_ci = price_quant5_model.predict(vehicle_features)

        # You can return any JSON-serializable value.
        return {
            "predicted_price": price_pred.tolist(),
            "upper_ci": upper_ci.tolist(),
            "lower_ci": lower_ci.tolist(),
        }
    except Exception as e:
        result = str(e)
        return result
