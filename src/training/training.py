# Author: Jonah Hamilton
# Date: 2023-08-16

import os, sys

import pandas as pd
import numpy as np
import json
import mlflow
from mlflow.models.signature import infer_signature
from dotenv import load_dotenv, find_dotenv
import pickle
import tempfile
import hydra
from omegaconf import DictConfig, OmegaConf
import hyperopt as hp

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MultiLabelBinarizer, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate, cross_val_score
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials
from azureml.core import Workspace
from azureml.core.model import Model


cur_dir = os.getcwd()
SRC_PATH = cur_dir[
    : cur_dir.index("fortunato-wheels-engine") + len("fortunato-wheels-engine")
]
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from src.data.car_ads import CarAds
from src.logs import get_logger
from src.data.training_preprocessing import preprocess_ads_for_training
from src.evaluate import price_model
from src.training.custom_components import MultiHotEncoder

load_dotenv(find_dotenv())

logger = get_logger(__name__)

AZURE_MLFLOW_URI = os.environ.get("AZURE_MLFLOW_URI")
mlflow.set_tracking_uri(AZURE_MLFLOW_URI)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def training(cfg: DictConfig) -> None:
    # Load the configuration file
    print(OmegaConf.to_yaml(cfg))

    # load the data
    ads = CarAds()
    # ads.get_car_ads(sources=["cargurus", "kijiji"])
    ads.get_car_ads(
        data_dump=os.path.join(
            SRC_PATH, "data", "processed", "car-ads-dump_2023-07-18.csv"
        )
    )

    model_features = OmegaConf.to_object(
        cfg.preprocess.model_feats.target
        + cfg.preprocess.model_feats.numeric
        + cfg.preprocess.model_feats.categorical
        + cfg.preprocess.model_feats.multi_label
    )

    # inital preprocessing
    ads.preprocess_ads(cfg.load_data.preprocess.top_n_options)

    # preprocess ads for training
    preprocessed_ads = preprocess_ads_for_training(
        ads.df, model_features=model_features,
                exclude_new_vehicle_ads=cfg.preprocess.exclude,
                min_num_ads=cfg.preprocess.min_num,
                max_age_at_posting=cfg.preprocess.max_age,
                min_price=cfg.preprocess.min_price,
                max_price=cfg.preprocess.max_price,
                )

    train_df, test_df = train_test_split(
        preprocessed_ads,
        test_size=cfg.preprocess.test_size,
        random_state=42,
        stratify=preprocessed_ads["model"],
    )

    # with features selected drop all with null values
    train_df = train_df[model_features].dropna().reset_index(drop=True)
    test_df = test_df[model_features].dropna().reset_index(drop=True)

    X_train = train_df.drop(columns=["price"])
    y_train = train_df["price"]
    X_test = test_df.drop(columns=["price"])
    y_test = test_df["price"]

    numeric = OmegaConf.to_object(cfg.preprocess.model_feats.numeric)
    categorical = OmegaConf.to_object(cfg.preprocess.model_feats.categorical)
    multi = OmegaConf.to_object(cfg.preprocess.model_feats.multi_label)

    # make column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("multi", MultiHotEncoder(), multi),
        ]
    )

    # hyperopt setup
    def objective(params, log_to_mlflow=True):
        classifier_type = params['type']
        del params['type']
        if classifier_type == 'gradient_boosting':
            clf = GradientBoostingRegressor(**params)
        elif classifier_type == 'xgboost':
            clf = XGBRegressor(**params)
        elif classifier_type == 'rf':
            clf = RandomForestRegressor(**params)
        elif classifier_type == 'ridge':
            clf = Ridge(**params)
        else:
            return 0

        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", clf),
            ]
        )

        metrics =  OmegaConf.to_object(cfg.hyperopt.metrics)
        # manually run cross_validate and get train/test rmse, mape, and r2
        model_cv_results = (
            pd.DataFrame(
                cross_validate(
                    pipe,
                    X_train.head(1000),
                    y_train.head(1000),
                    cv=5,
                    scoring=metrics, 
                    return_train_score=True,
                    n_jobs=-1,
                )
            )
            .agg(["mean", "std"])
            .T
        )

        if cfg.hyperopt.log_to_mlflow:
            # log metrics to mlflow
            with mlflow.start_run():
                # log train and test for each metric
                for m in metrics:
                    mlflow.log_metric(
                        f"{m}_train_mean", model_cv_results.loc[f"train_{m}"]["mean"]
                    )
                    mlflow.log_metric(
                        f"{m}_test_mean", model_cv_results.loc[f"test_{m}"]["mean"]
                    )
                    mlflow.log_metric(
                        f"{m}_train_std", model_cv_results.loc[f"train_{m}"]["std"]
                    )
                    mlflow.log_metric(f"{m}_test_std", model_cv_results.loc[f"test_{m}"]["std"])

                # log params
                mlflow.log_params(params)
                # log the type of model
                mlflow.log_param("model_type", classifier_type)

                fit_model = pipe.fit(X_train.head(1000), y_train.head(1000))

                # log model
                mlflow.sklearn.log_model(
                    fit_model,
                    "model",
                    signature=infer_signature(X_train.head(1000), y_train.head(1000)),
                )

                # predict on test set
                y_pred = fit_model.predict(X_test.head(1000))

                # add predicted price to test_df, round to 1 decimal place
                full_df = (
                    test_df.head(1000).copy(deep=True).assign(predicted_price=y_pred.round(1))
                )

                # calculate evaluation metrics by model
                metrics_by_model = price_model.calculate_evaluation_metrics_by_model(full_df)

                # calculate evaluation metrics by make
                metrics_by_make = price_model.calculate_evaluation_metrics_by_make(full_df)

                # calculate training data metrics
                train_data_metrics = price_model.calculate_train_data_metrics(
                    train_df.head(1000)
                )

                with tempfile.TemporaryDirectory() as tmpdir:
                    # Save model metrics to CSV file
                    model_metrics_fname = os.path.join(tmpdir, "metrics_by_model.csv")
                    metrics_by_model.to_csv(model_metrics_fname, index=False)

                    # Save make metrics to CSV file
                    make_metrics_fname = os.path.join(tmpdir, "metrics_by_make.csv")
                    metrics_by_make.to_csv(make_metrics_fname, index=False)

                    # Save train metrics to CSV file
                    train_metrics_fname = os.path.join(tmpdir, "train_data_metrics.csv")
                    train_data_metrics.to_csv(train_metrics_fname, index=False)

                    # Log metrics files as artifacts
                    mlflow.log_artifacts(tmpdir, artifact_path="evaluate/")

        else:
            fit_model = pipe.fit(X_train.head(1000), y_train.head(1000))
            y_pred = fit_model.predict(X_test.head(1000))
            full_df = (
                test_df.head(1000).copy(deep=True).assign(predicted_price=y_pred.round(1))
            )
            metrics_by_model = price_model.calculate_evaluation_metrics_by_model(full_df)
            metrics_by_make = price_model.calculate_evaluation_metrics_by_make(full_df)
            train_data_metrics = price_model.calculate_train_data_metrics(train_df.head(1000))

            # can either print out results or save as csv locally


        # make negative mape positive so it minimizes it
        result = {
            "loss": -model_cv_results.loc["test_" + metrics[0]]["mean"],
            "status": STATUS_OK,
        }

        return result


    # define search space
    search_space = hp.choice("classifier_type", [
        # {
        #     "type": "gradient_boosting",
        #     "max_features": hp.choice("max_features", ["sqrt", "log2"]),
        #     "max_depth": hp.uniformint("max_depth", 15, 30),
        #     "min_samples_split": hp.uniformint("dtree_min_samples_split", 20, 40),
        #     "n_estimators": hp.uniformint("n_estimators", 150, 300),
        # },
        {
            'type': 'xgboost',
            'max_depth': hp.uniformint('max_depth', 15, 40),
            'min_child_weight': hp.uniformint('min_child_weight', 1, 10),
            'subsample': hp.uniform('subsample', 0.5, 1),
            'n_estimators': hp.uniformint('n_estimators', 150, 400),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
            'gamma': hp.uniform('gamma', 0.1, 1),
        },
        # {
        #     'type': 'rf',
        #     'max_depth': hp.uniformint('max_depth', 5, 50),
        #     'max_features': hp.choice('max_features', ['sqrt', 'log2']),
        #     'min_samples_split': hp.uniform('min_samples_split', 0.1, 1),
        # },
        # {
        #     'type': 'ridge',
        #     'alpha': hp.uniform('alpha', 0.1, 100),
        # }
        ])

    # # Define the hyperopt search space based on the selected classifier type
    # hyperopt_space = cfg.search_space.gradient_boosting

    # # Define the hyperopt search space using the selected hyperparameters
    # search_space = {}
    # for key, value in hyperopt_space.items():
    #     search_space[key] = getattr(hp, value[0])(key, *value[1:])

   

    # mlflow.set_experiment("price-prediction-v3-gradboost")
    mlflow.set_experiment(cfg.mlflow.exp_name)
    mlflow.sklearn.autolog(disable=True)

    search_algorithm = tpe.suggest

    best_hyperparams = fmin(
        fn=objective,
        space=search_space,
        algo=search_algorithm,
        max_evals=cfg.mlflow.evals,
        trials=Trials(),
    )




if __name__ == "__main__":
    training()
