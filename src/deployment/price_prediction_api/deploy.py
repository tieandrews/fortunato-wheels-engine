# Author: Ty Andrews
# Date: 2023-06-04

import os
import sklearn
from azureml.core import Environment
from azureml.core import Workspace
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice
from azureml.core.model import Model
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# update model versions here
PRICE_PREDICTION_MODEL_VER = 3
PRICE_PREDICTION_MODEL_QUANT5_VER = 1
PRICE_PREDICTION_MODEL_QUANT95_VER = 1

CPU_CORES = 0.2
MEMORY_GB = 1.0


def main():
    logger.info("Loading workspace from config.json")
    ws = Workspace.from_config(
        # assumed running from root of repo
        path=os.path.join("src", "deployment", "price_prediction_api", "config.json")
    )
    print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep="\n")

    logger.info("Creating Azure environment")
    env = Environment("price-prediction-env")
    env.python.conda_dependencies.add_pip_package("joblib")
    env.python.conda_dependencies.add_pip_package("pandas==1.5")
    env.python.conda_dependencies.add_pip_package("numpy==1.23")
    env.python.conda_dependencies.add_pip_package("inference-schema")
    env.python.conda_dependencies.add_pip_package("azureml-defaults")
    env.python.conda_dependencies.add_pip_package(
        "scikit-learn=={}".format(sklearn.__version__)
    )

    inference_config = InferenceConfig(
        entry_script=os.path.join(
            "src", "deployment", "price_prediction_api", "score.py"
        ),
        environment=env,
    )

    logger.info(f"Loading models from workspace.")
    logger.info(
        f"Model Versions: price-prediction model ver: {PRICE_PREDICTION_MODEL_VER}, price-prediction-quant5 model ver: {PRICE_PREDICTION_MODEL_QUANT5_VER}, price-prediction-quant95 model ver: {PRICE_PREDICTION_MODEL_QUANT95_VER}"
    )
    price_model = Model(
        ws, "all-vehicles-price-prediction", version=PRICE_PREDICTION_MODEL_VER
    )
    price_model_quant5 = Model(
        ws,
        "all-vehicles-price-prediction-quant5",
        version=PRICE_PREDICTION_MODEL_QUANT5_VER,
    )
    price_model_quant95 = Model(
        ws,
        "all-vehicles-price-prediction-quant95",
        version=PRICE_PREDICTION_MODEL_QUANT95_VER,
    )

    logger.info(
        f"Creating deployment configuration with {CPU_CORES} CPU cores and {MEMORY_GB} GB memory"
    )
    deployment_config = AciWebservice.deploy_configuration(
        cpu_cores=CPU_CORES, memory_gb=MEMORY_GB, enable_app_insights=True
    )

    logger.info("Deploying model to Azure Container Instance (ACI)")
    service = Model.deploy(
        ws,
        "price-prediction-service",
        [price_model, price_model_quant5, price_model_quant95],
        inference_config,
        deployment_config,
        overwrite=True,
    )
    service.wait_for_deployment(show_output=True)

    logger.info("Deployment done, scoring URI: ", service.scoring_uri)


if __name__ == "__main__":
    main()
