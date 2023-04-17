# Author: Ty Andrews
# Date 2023-04-16

import os
import sys

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

cur_dir = os.getcwd()
src_path = cur_dir[
    : cur_dir.index("fortunato-wheels-engine") + len("fortunato-wheels-engine")
]
if src_path not in sys.path:
    sys.path.append(src_path)

from src.logs import get_logger

# Create a custom logger
logger = get_logger(__name__)

class CarPricePredictorLME:
    def __init__(
        self,
        target="price",
        fixed_continuous_features=list(),
        fixed_categorical_features=list(),
        random_effects=None,
        group=None,
        model_path=None,
    ):
        self.target = target
        self.fixed_continuous_features = fixed_continuous_features
        self.fixed_categorical_features = fixed_categorical_features
        self.random_effects = random_effects
        self.group = group

        if model_path:
            self.load_model(model_path)
        else:
            self.model = None

    def fit(self, data, display_summary=True):
        """ Fit the model to the data.

        Parameters
        ----------
        data : pd.DataFrame
            Data to fit the model to. Must contain all the features from the
            `fixed_continuous_features` and `fixed_categorical_features` attributes
            and the `target` attribute.
        display_summary : bool, optional
            Whether to display the model summary, by default True

        Raises
        ------
        AssertionError
            If the data does not contain all the required columns.

        Returns
        -------
        None
        """
        # check data contains all required columns
        assert all(
            [
                col in data.columns
                for col in self.fixed_continuous_features
                + self.fixed_categorical_features
                + [self.target]
            ]
        )

        formula = f"{self.target} ~ {' + '.join(self.fixed_continuous_features +  self.fixed_categorical_features)}"
        re_formula = (
            f"~ {' + '.join(self.random_effects)}" if self.random_effects else None
        )

        ols_md = smf.mixedlm(
            formula=formula,
            data=data,
            groups=self.group,
            re_formula=re_formula,  # random effects with different slopes and intercepts
        )

        self.model = ols_md.fit(method=["lbfgs"])

        if display_summary:
            print(self.model.summary())

    def evaluate_model(self, metrics=["rmse", "mape"]):
        """ Evaluate the model using the specified metrics.

        Parameters
        ----------
        metrics : list, optional
            List of metrics to evaluate the model with, by default ["rmse", "mape"]

        Raises
        ------
        Exception
            If the model has not been fitted yet.

        Returns
        -------
        dict
            Dictionary of the metrics and their values.
        """
        results = dict()
        if "rmse" in metrics:
            # calculate RMSE
            rmse = np.sqrt(
                mean_squared_error(self.model.fittedvalues, self.model.model.endog)
            )
            results["rmse"] = rmse
            logger.info(f"OLS Model price RMSE in CAD: ${rmse:.0f}")

        if "mape" in metrics:
            # calculate MAPE
            mape = np.mean(
                np.abs(
                    (self.model.fittedvalues - self.model.model.endog)
                    / self.model.model.endog
                )
            )
            results["mape"] = mape
            logger.info(f"OLS Model price MAPE: {mape:.2%}")

        return results

    def predict(self, X):
        """ Predict the price of the cars in the given data.

        Parameters
        ----------
        X : pd.DataFrame
            Data to predict the price of. Must contain all the features from the
            `fixed_continuous_features` and `fixed_categorical_features` attributes.

        Raises
        ------
        Exception
            If the model has not been fitted yet.
        Exception
            If the data does not contain all the required columns.

        Returns
        -------
        pd.Series
            Series of the predicted prices.
        """
        if self.model is None:
            raise Exception("Model not fitted yet")
        elif not isinstance(X, pd.DataFrame):
            raise Exception("X must be a pandas DataFrame")
        elif not all(
            [
                col in X.columns
                for col in self.fixed_continuous_features
                + self.fixed_categorical_features
                + [self.group]
            ]
        ):
            raise Exception("X must contain all required columns")

        # create results series size of X initialized with all zeros named "pred_price"
        results = pd.Series(np.zeros(X.shape[0]), name="pred_price")

        # add intercept to pred_price
        results += self.model.params["Intercept"]

        # add continuous fixed effects to pred_price
        for feat in self.fixed_continuous_features:
            results += (
                self.model.params[feat] * X[feat]
            ).to_numpy()  # need this otherwise adding series doesn't work

        # add random effects and categorical effects to pred_price
        for i, car in X.reset_index(drop=True).iterrows():
            for feat in self.random_effects:
                if "wheel_system" in feat:
                    # 4WD is the baseline of the model so no contribution, just base intercept
                    if car["wheel_system"] in ["4WD", "RWD"]:
                        continue
                    results.iloc[i] += self.model.random_effects[car[self.group]][
                        f"wheel_system[T.{car['wheel_system']}]"
                    ]
                else:
                    results.iloc[i] += (
                        self.model.random_effects[car[self.group]][feat] * car[feat]
                    )

            for feat in self.fixed_categorical_features:
                if car["wheel_system"] in ["4WD", "RWD"]:
                    continue
                # results.iloc[i] += self.model.random_effects[car[self.group]][f"wheel_system[T.{car['wheel_system']}]"]
                results.iloc[i] += self.model.params[f"{feat}[T.{car[feat]}]"]

            # add grouping effects to pred_price
            results.iloc[i] += self.model.random_effects[car[self.group]][self.group]

        return results

    def save_model(self, path):
        """ Save the model to the specified path.
        
        Parameters
        ----------
        path : str
            Path to save the model to including desired file name.

        Returns
        -------
        None
        """
        self.model.save(path)

    def load_model(self, path):
        """ Load the model from the specified path.

        Parses the statsmodel objects parameters to get the fixed and random effects
        as well as the group category if applicable.

        Parameters
        ----------
        path : str
            Path to load the model from including file name.

        Returns
        -------
        None
        """
        self.model = sm.load(path)

        continuous_features = [
            col
            for col in m.model.fe_params.index
            if (col not in m.fixed_categorical_features)
            and (col != "Intercept")
            and ("[T" not in col)
        ]
        self.fixed_continuous_features = continuous_features

        categorical_features = list(
            set(
                [
                    col.split("[T")[0]
                    for col in m.model.fe_params.index
                    if (col in m.fixed_categorical_features) or ("[T" in col)
                ]
            )
        )
        self.fixed_categorical_features = categorical_features

        # the first random effect is the grouping effect
        group_feature = list(m.model.random_effects.items())[0][1].index[0]
        self.group = group_feature

        # remaining random effects are the random effects
        random_effects = list((list(m.model.random_effects.items())[0][1].index[1:]))
        self.random_effects = random_effects
