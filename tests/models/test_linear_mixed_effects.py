import os
import sys

import pytest

SRC_PATH = sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from src.models.linear_mixed_effects import CarPricePredictorLME


# check basic creation of the class returns a valid object
@pytest.mark.parametrize(
    "continuous_features,categorical_features,model_path",
    [
        (["mileage"], [], None),
        ([], ["trim"], None),
        # (
        #     [],
        #     [],
        #     os.path.join("data", "testing", "test_suv-price-model-v1.pkl"),
        # ),
    ],
)
def test_car_price_predictor_lme(continuous_features, categorical_features, model_path):
    predictor = CarPricePredictorLME(
        fixed_continuous_features=continuous_features,
        fixed_categorical_features=categorical_features,
        model_path=model_path,
    )

    assert isinstance(predictor, CarPricePredictorLME)


# check that initializing the class with no fixed effects or model path raises
# an assertion error
def test_check_invalid_price_predictor_lme():
    with pytest.raises(AssertionError):
        predictor = CarPricePredictorLME(
            fixed_continuous_features=[], fixed_categorical_features=[], model_path=None
        )


# must have a grouping variable by string for the random effects
def test_invalid_group_param():
    with pytest.raises(AssertionError):
        predictor = CarPricePredictorLME(
            fixed_continuous_features=["mileage"],
            fixed_categorical_features=["trim"],
            group=None,
        )
